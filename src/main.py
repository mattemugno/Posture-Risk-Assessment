import io
import logging
import os
import threading

from flask import Flask, Response, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from src.PoseDetector import PoseDetector
from src.PostureEvaluation import PostureEvaluation
from src.utils import get_image_zoomed

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
template_folder = os.path.join(base_dir, 'template')
data_path = os.path.join(base_dir, 'data', 'data.json')

app = Flask(__name__, template_folder=template_folder)

route_lock = threading.Lock()


class PoseManager:
    def __init__(self):
        self.pose = None
        self.posture = None
        self.recording = False

    def create_pose_detector(self):
        if self.pose is None:
            self.pose = PoseDetector(model_complexity=0)

    def destroy_pose_detector(self):
        if self.pose is not None:
            del self.pose
            self.pose = None

    def create_posture_eval(self):
        if self.posture is None:
            self.posture = PostureEvaluation(data_path=data_path)

    def destroy_posture_eval(self):
        if self.posture is not None:
            del self.posture
            self.posture = None

    def reset_posture_var(self):
        if self.posture is not None:
            self.posture.not_acceptable_shldr_r = 0
            self.posture.not_acceptable_shldr_l = 0
            self.posture.not_acceptable_count_trunk = 0


pose_manager = PoseManager()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    if request.method == 'POST':
        if request.form.get('segmask') == 'Segmentation Mask':
            if pose_manager.pose.enable_segmentation:
                pose_manager.pose.enable_segmentation = False
            else:
                pose_manager.pose.enable_segmentation = True
        elif request.form.get('stop') == 'Open/Close Video Camera':
            if pose_manager.recording:
                pose_manager.destroy_pose_detector()
                pose_manager.recording = False
            else:
                pose_manager.create_pose_detector()
                pose_manager.recording = True
        elif request.form.get('stop') == 'Start/Stop Recording':
            if pose_manager.pose.record_session:
                pose_manager.pose.record_session = False
                logging.info('Stop Recording')
            else:
                pose_manager.pose.record_session = True
                logging.info('Start Recording')
        elif request.form.get('save') == 'Save Data':
            pose_manager.pose.save_data()
    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/show_all_plots')
def show_all_plots():
    pose_manager.create_posture_eval()
    return render_template('show_all_plots.html')


@app.route('/plot')
def plot_posture_in_time():
    fig, axs = pose_manager.posture.plot_angles_over_time()

    '''ax = axs[0]
    get_image_zoomed(ax)

    ax = axs[2]
    get_image_zoomed(ax)'''

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/postureStatus')
def plot_posture_status():
    fig = pose_manager.posture.plot_posture_status()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/piecharts')
def plot_pie_charts():
    pose_manager.reset_posture_var()
    pose_manager.posture.calculate_percentages()
    fig = pose_manager.posture.plot_pie_charts()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


@app.route('/plot_heatmap')
def plot_heatmap():
    fig = pose_manager.posture.plot_heatmap()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')


def gen():
    while True:
        if pose_manager.pose is None:
            image_path = os.path.join(base_dir, 'img', 'video_player_placeholder.png')
            with open(image_path, 'rb') as img_file:
                frame = img_file.read()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            frame = pose_manager.pose.find_landmarks()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
