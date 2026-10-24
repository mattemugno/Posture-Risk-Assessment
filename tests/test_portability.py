import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PortabilityTests(unittest.TestCase):
    def test_bundled_mediapipe_model_runs_without_a_camera(self):
        import mediapipe as mp
        import numpy as np

        # Complexity 1 uses the bundled model, without downloading camera assets.
        with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=1) as model:
            results = model.process(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertIsNone(results.pose_landmarks)

    def test_app_from_another_directory_with_unicode_paths_and_no_gui(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='posture portabilità ') as directory:
            root = Path(directory)
            project = root / 'demo con spazi'
            project.mkdir()
            for folder in ('src', 'template', 'static', 'img'):
                shutil.copytree(source / folder, project / folder,
                                ignore=shutil.ignore_patterns('__pycache__'))
            (project / 'data').mkdir()
            records = [
                {'timestamp': index, 'r_shldr': 5, 'l_shldr': 10, 'trunk_incl': 5,
                 'note': 'Sessione di prova: qualità'}
                for index in range(2)
            ]
            (project / 'data' / 'sessione qualità.json').write_text(
                json.dumps(records, ensure_ascii=False), encoding='utf-8')
            script = '''
import sys
sys.path.insert(0, sys.argv[1])
from concurrent.futures import ThreadPoolExecutor
from src.main import app, pose_manager
from src.video_overlay import _font
import matplotlib

assert matplotlib.get_backend().lower() == 'agg'
assert _font(16).getlength('12.34°') > 0
assert _font(16, bold=True).getlength('12.34°') > 0
client = app.test_client()
assert client.get('/').status_code == 200
assert client.get('/static/css/base.css').status_code == 200
assert client.get('/get_json_files').get_json()['json_files'] == ['sessione qualità.json']
selection = {'json_file': 'sessione qualità.json'}
assert client.post('/show_all_plots', data=selection).status_code == 200
assert client.get('/session_data', query_string=selection).get_json()['timestamps'] == [0, 1]

def render_charts():
    with app.test_client() as threaded_client:
        for route in ('/plot', '/postureStatus', '/plot_heatmap', '/piecharts'):
            response = threaded_client.get(route, query_string=selection)
            assert response.status_code == 200, (route, response.status_code)
            assert response.data.startswith(bytes.fromhex('89504e470d0a1a0a')), route

with ThreadPoolExecutor(max_workers=1) as executor:
    executor.submit(render_charts).result(timeout=30)
assert pose_manager.pose is None
'''
            environment = dict(os.environ, MPLBACKEND='MacOSX',
                               MPLCONFIGDIR=str(root / 'matplotlib'),
                               PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
            result = subprocess.run(
                [sys.executable, '-B', '-c', script, str(project)], cwd=root,
                env=environment, capture_output=True, encoding='utf-8', timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
