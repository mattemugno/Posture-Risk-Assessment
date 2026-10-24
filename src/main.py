import io
import logging
import os
import time
from functools import wraps

from flask import Flask, Response, abort, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from src.PoseManager import PoseManager
from src.config import base_dir, data_path, template_folder
from src.utils import set_logger

set_logger()

app = Flask(__name__, template_folder=template_folder,
            static_folder=os.path.join(base_dir, 'static'))

pose_manager = PoseManager()


@app.context_processor
def presentation_state():
    with pose_manager.lock:
        pose = pose_manager.pose
        return {'camera_open': pose is not None,
                'session_active': pose is not None and pose.record_session}


def synchronized(lock_name):
    def decorate(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            with getattr(pose_manager, lock_name):
                return view(*args, **kwargs)
        return wrapped
    return decorate


def get_posture(allow_empty=False):
    filename = request.args.get('json_file')
    if filename:
        try:
            pose_manager.create_posture_eval(filename)
        except (OSError, ValueError, KeyError, TypeError):
            abort(400, description='Unable to load the selected session.')
    if pose_manager.posture is None:
        abort(400, description='Select a session first.')
    if not allow_empty and not pose_manager.posture.time:
        abort(400, description='The selected session has no complete posture samples.')
    return pose_manager.posture


def png_response(figure):
    """Serialize a chart using the same PNG response for every report route."""
    output = io.BytesIO()
    FigureCanvas(figure).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/requests', methods=['POST', 'GET'])
@synchronized('lock')
def tasks():
    if request.method == 'POST':
        if pose_manager.pose is None and (
                request.form.get('segmask') == 'Segmentation Mask'
                or request.form.get('session') in ('Start Session', 'Stop Session')):
            return render_template('index.html', error='Open the camera before starting a session or segmentation.'), 409
        try:
            if request.form.get('segmask') == 'Segmentation Mask':
                pose_manager.pose.enable_segmentation = not pose_manager.pose.enable_segmentation
            elif request.form.get('stop') == 'Open Video Camera':
                pose_manager.create_pose_detector()
            elif request.form.get('stop') == 'Close Video Camera':
                pose_manager.destroy_pose_detector()
            elif request.form.get('session') == 'Stop Session':
                if pose_manager.pose.record_session:
                    pose_manager.pose.save_data()
                    pose_manager.pose.record_session = False
                    logging.info('Stop Session')
            elif request.form.get('session') == 'Start Session':
                if not pose_manager.pose.record_session:
                    pose_manager.pose.data = []
                    pose_manager.pose.record_session = True
                    logging.info('Start Session')
        except (OSError, RuntimeError) as error:
            logging.warning('Camera/session operation failed: %s', error)
            return render_template('index.html', error=str(error)), 503
    return render_template('index.html')


@app.route('/show_all_plots', methods=['GET', 'POST'])
@synchronized('posture_lock')
def show_all_plots():
    filename = request.form.get('json_file')
    if filename:
        try:
            pose_manager.create_posture_eval(filename)
        except (OSError, ValueError, KeyError, TypeError):
            return render_template('index.html', error='Unable to load the selected session.'), 400
    if pose_manager.posture is None:
        return render_template('index.html', error='Select a session first.'), 400
    return render_template('show_all_plots.html',
                           json_file=os.path.basename(pose_manager.posture.data_path),
                           has_samples=bool(pose_manager.posture.time))


@app.route('/get_json_files', methods=['GET'])
def get_json_files():
    json_files = [f for f in os.listdir(data_path) if f.endswith('.json')]
    return {'json_files': json_files}


@app.route('/session_data')
@synchronized('posture_lock')
def session_data():
    if not request.args.get('json_file'):
        abort(400, description='Select a session first.')
    posture = get_posture(allow_empty=True)
    try:
        return posture.to_chart_data()
    except (ValueError, TypeError, OverflowError):
        abort(400, description='Unable to read chart values from the selected session.')


@app.route('/plot')
@synchronized('posture_lock')
def plot_posture_in_time():
    figure, _ = get_posture().plot_angles_over_time()
    return png_response(figure)


@app.route('/postureStatus')
@synchronized('posture_lock')
def plot_posture_status():
    return png_response(get_posture().plot_posture_status())


@app.route('/piecharts')
@synchronized('posture_lock')
def plot_pie_charts():
    posture = get_posture()
    pose_manager.reset_posture_var()
    posture.calculate_percentages()
    return png_response(posture.plot_pie_charts())


@app.route('/plot_heatmap')
@synchronized('posture_lock')
def plot_heatmap():
    return png_response(get_posture().plot_heatmap())


def gen():
    image_path = os.path.join(base_dir, 'img', 'video_player_placeholder.png')
    with open(image_path, 'rb') as img_file:
        placeholder = img_file.read()
    while True:
        frame = pose_manager.get_frame()
        # The bundled placeholder is a GIF despite its .png filename.
        content_type = b'image/png' if frame is not None else b'image/gif'
        yield (b'--frame\r\nContent-Type: ' + content_type + b'\r\n\r\n'
               + (frame if frame is not None else placeholder) + b'\r\n\r\n')
        time.sleep(0.03 if frame is not None else 0.1)


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
