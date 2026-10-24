import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

from matplotlib.figure import Figure

from src.main import app
from src.PoseDetector import PoseDetector
from src.PoseManager import PoseManager
from src.VideoCamera import VideoCamera


class FakeDetector:
    save_data = PoseDetector.save_data

    def __init__(self, directory):
        self.data_path = directory
        self.data = []
        self.record_session = False
        self.enable_segmentation = False
        self.recorded = Event()
        self.read = Event()
        self.closed = False
        self.marker = 1
        self.frame_count = 0
        self.reads_after_close = 0

    def find_landmarks(self):
        time.sleep(0.002)
        if self.closed:
            self.reads_after_close += 1
        self.frame_count += 1
        if self.record_session:
            self.data.append({'session': self.marker})
            self.recorded.set()
        self.read.set()
        return b'camera-frame'

    def close(self):
        self.closed = True


class ApplicationSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manager = PoseManager()
        self.addCleanup(patch.stopall)
        patch('src.main.pose_manager', self.manager).start()
        patch('src.PoseManager.PoseDetector',
              side_effect=lambda **kwargs: FakeDetector(self.directory.name)).start()
        app.config['TESTING'] = True
        self.client = app.test_client()

    def tearDown(self):
        self.manager.destroy_pose_detector()
        if self.manager._capture_thread is not None:
            self.manager._capture_thread.join(timeout=1)
            self.assertFalse(self.manager._capture_thread.is_alive())

    def open_camera(self):
        response = self.client.post('/requests', data={'stop': 'Open Video Camera'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.manager.pose.read.wait(1))
        return self.manager.pose

    def saved_sessions(self):
        return [json.loads(path.read_text()) for path in Path(self.directory.name).glob('*.json')]

    def test_controls_and_stream_without_camera(self):
        for command in [{'session': 'Start Session'}, {'segmask': 'Segmentation Mask'}]:
            response = self.client.post('/requests', data=command)
            self.assertEqual(response.status_code, 409)
            self.assertIn(b'role="alert"', response.data)
        response = self.client.get('/video_feed', buffered=False)
        self.assertEqual(response.status_code, 200)
        frame = next(response.response)
        self.assertTrue(frame.startswith(b'--frame\r\nContent-Type: image/gif\r\n\r\nGIF89a'))
        response.close()

    def assert_controls(self, response, camera, session):
        html = response.get_data(as_text=True)
        self.assertRegex(html, f'id="cameraButton"[^>]*value="{camera}"')
        self.assertRegex(html, f'id="sessionButton"[^>]*value="{session}"')

    def test_controls_follow_server_state_across_reloads(self):
        self.assert_controls(self.client.get('/'), 'Open Video Camera', 'Start Session')
        self.open_camera()
        self.assert_controls(self.client.get('/'), 'Close Video Camera', 'Start Session')
        response = self.client.post('/requests', data={'session': 'Start Session'})
        self.assert_controls(response, 'Close Video Camera', 'Stop Session')
        self.assert_controls(self.client.get('/'), 'Close Video Camera', 'Stop Session')
        response = self.client.post('/requests', data={'stop': 'Close Video Camera'})
        self.assert_controls(response, 'Open Video Camera', 'Start Session')

    def test_repeated_commands_do_not_toggle_or_reset_recordings(self):
        pose = self.open_camera()
        self.client.post('/requests', data={'stop': 'Open Video Camera'})
        self.assertIs(self.manager.pose, pose)
        self.client.post('/requests', data={'session': 'Start Session'})
        self.assertTrue(pose.recorded.wait(1))
        recorded_data = pose.data
        self.client.post('/requests', data={'session': 'Start Session'})
        self.assertIs(pose.data, recorded_data)
        self.assertTrue(pose.record_session)
        for _ in range(2):
            self.client.post('/requests', data={'session': 'Stop Session'})
        self.assertFalse(pose.record_session)
        self.assertEqual(len(self.saved_sessions()), 1)
        for _ in range(2):
            self.client.post('/requests', data={'stop': 'Close Video Camera'})
        self.assertIsNone(self.manager.pose)

    def test_unavailable_camera_does_not_load_model_or_leave_state(self):
        camera = Mock()
        camera.isOpen.return_value = False
        with patch('src.PoseManager.PoseDetector', PoseDetector), \
                patch('src.PoseDetector.VideoCamera', return_value=camera), \
                patch('src.PoseDetector.mp.solutions.holistic.Holistic') as model:
            response = self.client.post('/requests', data={'stop': 'Open Video Camera'})
        self.assertEqual(response.status_code, 503)
        self.assertIn(b'No camera available', response.data)
        self.assertIsNone(self.manager.pose)
        self.assertFalse(self.manager.recording)
        self.assert_controls(response, 'Open Video Camera', 'Start Session')
        camera.close.assert_called_once()
        model.assert_not_called()

    def test_recording_continues_without_video_clients(self):
        pose = self.open_camera()
        self.client.post('/requests', data={'session': 'Start Session'})
        self.assertTrue(pose.recorded.wait(1))
        with self.manager.lock:
            before = len(pose.data)
            pose.recorded.clear()
        self.assertTrue(pose.recorded.wait(1))
        with self.manager.lock:
            self.assertGreater(len(pose.data), before)

    def test_rendering_statistics_does_not_block_capture(self):
        pose = self.open_camera()
        started = Event()
        finish = Event()

        def plot():
            started.set()
            if not finish.wait(2):
                raise RuntimeError('Chart test timed out')
            return Figure(), []

        self.manager.posture = SimpleNamespace(time=[0], plot_angles_over_time=plot)

        def request_chart():
            with app.test_client() as client:
                return client.get('/plot').status_code

        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(request_chart)
            try:
                self.assertTrue(started.wait(1))
                with self.manager.lock:
                    pose.read.clear()
                self.assertTrue(pose.read.wait(1))
            finally:
                finish.set()
            self.assertEqual(result.result(timeout=2), 200)

    def test_sessions_do_not_include_previous_samples(self):
        pose = self.open_camera()
        for marker in [1, 2]:
            with self.manager.lock:
                pose.marker = marker
                pose.recorded.clear()
            self.client.post('/requests', data={'session': 'Start Session'})
            self.assertTrue(pose.recorded.wait(1))
            response = self.client.post('/requests', data={'session': 'Stop Session'})
            self.assertEqual(response.status_code, 200)
        sessions = self.saved_sessions()
        self.assertEqual(len(sessions), 2)
        self.assertEqual({tuple({row['session'] for row in rows}) for rows in sessions}, {(1,), (2,)})

    def test_closing_camera_saves_and_stops_active_recording(self):
        pose = self.open_camera()
        self.client.post('/requests', data={'session': 'Start Session'})
        self.assertTrue(pose.recorded.wait(1))
        response = self.client.post('/requests', data={'stop': 'Close Video Camera'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.saved_sessions()[0])
        self.assertIsNone(self.manager.pose)
        self.assertFalse(self.manager.recording)
        self.assertTrue(pose.closed)
        self.assertEqual(pose.reads_after_close, 0)

    def test_save_failure_preserves_active_session(self):
        pose = self.open_camera()
        self.client.post('/requests', data={'session': 'Start Session'})
        self.assertTrue(pose.recorded.wait(1))
        with patch.object(pose, 'save_data', side_effect=OSError('Save failed')):
            for command in [{'session': 'Stop Session'}, {'stop': 'Close Video Camera'}]:
                response = self.client.post('/requests', data=command)
                self.assertEqual(response.status_code, 503)
                self.assertIs(self.manager.pose, pose)
                self.assertTrue(pose.record_session)
                self.assertFalse(pose.closed)
                self.assert_controls(response, 'Close Video Camera', 'Stop Session')

    def test_multiple_streams_share_capture_and_survive_camera_close(self):
        pose = self.open_camera()
        with self.manager.lock:
            before = pose.frame_count
            streams = [self.client.get('/video_feed', buffered=False) for _ in range(2)]
            for stream in streams:
                self.assertIn(b'camera-frame', next(stream.response))
            self.assertEqual(pose.frame_count, before)
        self.client.post('/requests', data={'stop': 'Close Video Camera'})
        for stream in streams:
            self.assertIn(b'image/gif', next(stream.response))
            stream.close()


class FrameAndStorageTests(unittest.TestCase):
    def test_failed_frame_read_is_ignored(self):
        pose = PoseDetector.__new__(PoseDetector)
        pose.cap = SimpleNamespace(isOpen=lambda: True, get_frame=lambda _: (None, None))
        pose.holistic = object()
        self.assertIsNone(pose.find_landmarks())
        camera = VideoCamera.__new__(VideoCamera)
        camera.video = Mock()
        camera.video.read.return_value = (True, None)
        model = Mock()
        self.assertEqual(camera.get_frame(model), (None, None))
        model.process.assert_not_called()

    def test_same_time_saves_are_distinct_and_respect_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            pose = PoseDetector.__new__(PoseDetector)
            pose.data_path = directory
            with patch('src.PoseDetector.datetime') as clock:
                clock.now.return_value = datetime(2026, 10, 21, 10, 0, 0)
                for marker in [1, 2]:
                    pose.data = [{'session': marker}]
                    pose.save_data()
            paths = list(Path(directory).glob('*.json'))
            self.assertEqual(len(paths), 2)
            self.assertEqual({json.loads(path.read_text())[0]['session'] for path in paths}, {1, 2})
            destination = Path(directory, 'custom')
            destination.mkdir()
            pose.save_data(destination)
            self.assertEqual(len(list(destination.glob('*.json'))), 1)


class StatisticsSafetyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.manager = PoseManager()
        self.addCleanup(patch.stopall)
        patch('src.main.pose_manager', self.manager).start()
        patch('src.PostureEvaluation.data_path', str(self.data)).start()
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        for name, angle in [('first.json', 5), ('second.json', 45)]:
            records = [{'timestamp': i, 'l_shldr': angle, 'r_shldr': angle, 'trunk_incl': 5}
                       for i in range(2)]
            (self.data / name).write_text(json.dumps(records))
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_traversal_and_absolute_paths_are_rejected(self):
        self.client.post('/show_all_plots', data={'json_file': 'first.json'})
        previous = self.manager.posture
        outside = self.root / 'outside.json'
        outside.write_text((self.data / 'first.json').read_text())
        for filename in ['../outside.json', str(outside), '../README.md']:
            response = self.client.post('/show_all_plots', data={'json_file': filename})
            self.assertEqual(response.status_code, 400)
            self.assertIs(self.manager.posture, previous)
            self.assertEqual(self.client.get('/plot', query_string={'json_file': filename}).status_code, 400)

    def test_symlink_outside_data_is_rejected(self):
        outside = self.root / 'outside.json'
        outside.write_text((self.data / 'first.json').read_text())
        try:
            (self.data / 'alias.json').symlink_to(outside)
        except (OSError, NotImplementedError) as error:
            self.skipTest(f'Symlink creation is unavailable: {error}')
        self.assertEqual(self.client.post('/show_all_plots', data={'json_file': 'alias.json'}).status_code, 400)
        self.assertEqual(self.client.get('/plot?json_file=alias.json').status_code, 400)

    def test_chart_stays_on_its_selected_file_after_other_client_selects(self):
        response = self.client.post('/show_all_plots', data={'json_file': 'first.json'})
        self.assertIn(b'json_file=first.json', response.data)
        with app.test_client() as other:
            other.post('/show_all_plots', data={'json_file': 'second.json'})
        response = self.client.get('/plot?json_file=first.json')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data.startswith(b'\x89PNG\r\n\x1a\n'))
        self.assertEqual(self.manager.posture.l_shldr, [5, 5])

    def test_statistics_remain_available_without_camera(self):
        response = self.client.post('/show_all_plots', data={'json_file': 'first.json'})
        self.assertEqual(response.status_code, 200)
        selected = self.manager.posture
        self.assertEqual(self.client.get('/show_all_plots').status_code, 200)
        self.assertIs(self.manager.posture, selected)
        for route in ['/plot', '/postureStatus', '/plot_heatmap', '/piecharts']:
            response = self.client.get(route + '?json_file=first.json')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.mimetype, 'image/png')
        self.assertIsNone(self.manager.pose)


if __name__ == '__main__':
    unittest.main()
