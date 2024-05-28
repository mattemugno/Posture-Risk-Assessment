import io
import logging
import os

from flask import Flask, Response, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from src.PoseManager import PoseManager
from src.config import template_folder, base_dir

app = Flask(__name__, template_folder=template_folder)

pose_manager = PoseManager()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    if request.method == 'POST':
        request.form.get('stop')
        if request.form.get('segmask') == 'Segmentation Mask':
            if pose_manager.pose.enable_segmentation:
                pose_manager.pose.enable_segmentation = False
            else:
                pose_manager.pose.enable_segmentation = True
        elif request.form.get('stop') == 'Open Video Camera' or request.form.get('stop') == 'Close Video Camera':
            if pose_manager.recording:
                logging.info('Stop video camera')
                pose_manager.destroy_pose_detector()
                pose_manager.recording = False
            else:
                logging.info('Open video camera')
                pose_manager.create_pose_detector()
                pose_manager.recording = True
        elif request.form.get('session') == 'Start Session' or request.form.get('session') == 'Stop Session':
            if pose_manager.pose.record_session:
                pose_manager.pose.record_session = False
                pose_manager.pose.save_data()
                logging.info('Stop Session')
            else:
                pose_manager.pose.record_session = True
                logging.info('Start Session')
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

    # plot shoulder
    '''ax = axs[0]
    get_image_zoomed(ax)'''

    # plot trunk
    '''ax = axs[2]
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
