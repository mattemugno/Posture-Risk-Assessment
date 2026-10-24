import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.main import app
from src.PoseManager import PoseManager
from src.PostureEvaluation import PostureEvaluation


class InteractiveChartTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.manager = PoseManager()
        for patcher in (
            patch('src.main.pose_manager', self.manager),
            patch('src.PostureEvaluation.data_path', str(self.data)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.records = [
            {'timestamp': 1000 + index * 1.125, 'r_shldr': right,
             'l_shldr': 0, 'trunk_incl': trunk}
            for index, (right, trunk) in enumerate(zip(
                [0] + [30] * 21 + [70, 0],
                [0] + [15] * 21 + [25, 0],
            ))
        ]
        self.write_session('first.json', self.records)

    def write_session(self, filename, records):
        (self.data / filename).write_text(json.dumps(records))

    def get_data(self, filename='first.json'):
        return self.client.get('/session_data', query_string={'json_file': filename})

    def test_payload_preserves_samples_status_and_existing_report_percentages(self):
        missing = {'timestamp': 2000, 'r_shldr': None, 'l_shldr': 0, 'trunk_incl': 0}
        self.write_session('first.json', self.records + [missing])
        response = self.get_data()
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload['timestamps'], [row['timestamp'] for row in self.records])
        self.assertEqual(payload['series'][0]['angles'], [row['r_shldr'] for row in self.records])
        self.assertEqual(payload['series'][1]['angles'], [row['l_shldr'] for row in self.records])
        self.assertEqual(payload['series'][2]['angles'], [row['trunk_incl'] for row in self.records])
        expected_status = [0] + [1] * 20 + [2, 2, 0]
        self.assertEqual(payload['series'][0]['status'], expected_status)
        self.assertEqual(payload['series'][1]['status'], [0] * 24)
        self.assertEqual(payload['series'][2]['status'], expected_status)

        figure = self.manager.posture.plot_pie_charts()
        for index, item in enumerate(payload['risk']):
            axis = figure.axes[2 * index]
            self.assertEqual(item['label'], axis.get_title())
            bad_wedge = axis.patches[1]
            self.assertAlmostEqual(item['not_acceptable'],
                                   (bad_wedge.theta2 - bad_wedge.theta1) / 360 * 100, places=4)
            self.assertAlmostEqual(item['acceptable'] + item['not_acceptable'], 100)
        self.assertIsNone(self.manager.pose)

    def test_each_request_stays_on_its_selected_session(self):
        self.write_session('second.json', [dict(row, r_shldr=90) for row in self.records])
        self.get_data('second.json')
        first = self.get_data().get_json()
        self.assertEqual(first['series'][0]['angles'][0], 0)
        second = self.get_data('second.json').get_json()
        self.assertEqual(second['series'][0]['angles'][0], 90)
        self.assertEqual(self.get_data().get_json(), first)

    def test_file_selection_is_required_and_paths_are_restricted(self):
        self.assertEqual(self.client.get('/session_data').status_code, 400)
        outside = self.root / 'outside.json'
        outside.write_text(json.dumps(self.records))
        (self.data / 'invalid.json').write_text('not JSON')
        for filename in ('../outside.json', str(outside), 'invalid.json', 'missing.json'):
            with self.subTest(filename=filename):
                self.assertEqual(self.get_data(filename).status_code, 400)

    def test_symlink_outside_data_is_rejected(self):
        outside = self.root / 'outside.json'
        outside.write_text(json.dumps(self.records))
        try:
            (self.data / 'link.json').symlink_to(outside)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f'Symlink creation is unavailable: {error}')
        self.assertEqual(self.get_data('link.json').status_code, 400)

    def test_empty_session_has_no_risk_calculation(self):
        self.write_session('empty.json', [])
        response = self.get_data('empty.json')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['timestamps'], [])
        self.assertEqual(response.get_json()['risk'], [])

    def test_non_finite_numbers_do_not_reach_the_browser(self):
        for value in (float('nan'), float('inf'), 'invalid'):
            self.write_session('invalid-values.json', [dict(self.records[0], r_shldr=value)])
            self.assertEqual(self.get_data('invalid-values.json').status_code, 400)

    def test_empty_page_does_not_request_legacy_png_reports(self):
        self.write_session('empty.json', [])
        response = self.client.post('/show_all_plots', data={'json_file': 'empty.json'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'no complete posture samples', response.data)
        for endpoint in ('/plot?', '/postureStatus?', '/plot_heatmap?', '/piecharts?'):
            self.assertNotIn(endpoint.encode(), response.data)

    def test_legacy_reports_reject_sessions_without_complete_samples(self):
        for records in ([], [dict(self.records[0], r_shldr=None)]):
            self.write_session('empty.json', records)
            for endpoint in ('/plot', '/postureStatus', '/plot_heatmap', '/piecharts'):
                with self.subTest(endpoint=endpoint, records=records):
                    response = self.client.get(endpoint, query_string={'json_file': 'empty.json'})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(b'no complete posture samples', response.data)

    def test_serializing_twice_does_not_accumulate_risk_counts(self):
        posture = PostureEvaluation('first.json')
        first = posture.to_chart_data()
        self.assertEqual(posture.to_chart_data(), first)
        self.assertEqual(posture.not_acceptable_shldr_r, 2)
        self.assertEqual(posture.not_acceptable_count_trunk, 2)

    def test_page_uses_local_bundle_and_retains_static_reports(self):
        response = self.client.post('/show_all_plots', data={'json_file': 'first.json'})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'/static/vendor/plotly-cartesian-4.1.1.min.js', response.data)
        self.assertIn(b'/session_data?json_file=first.json', response.data)
        self.assertEqual(response.data.count(b'class="chart-fallback"'), 4)
        self.assertIn(b'id="range-form"', response.data)


if __name__ == '__main__':
    unittest.main()
