import os
import threading

from flask import Flask, Response, render_template, request

from src.PoseDetector import PoseDetector
from src.PostureEvaluation import PostureEvaluation
from src.utils import plot_pie_charts_shoulders

device_id = 0
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
template_folder = os.path.join(base_dir, 'template')

app = Flask(__name__, template_folder=template_folder)
pose = PoseDetector(enable_face_landmarks=False)


@app.route('/')
def index():
    """
    Video streaming home page.
    """
    return render_template('index.html')


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    if request.method == 'POST':
        if request.form.get('segmask') == 'Segmentation Mask':
            if pose.enable_segmentation:
                pose.enable_segmentation = False
            else:
                pose.enable_segmentation = True
        elif request.form.get('stop') == 'Stop/Start':
            if pose.cap.recording:
                pose.cap.recording = False
            else:
                pose.cap.recording = True
        elif request.form.get('save') == 'Save Data':
            pose.save_data()
    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/plot')
def plot_posture_in_time():
    return Response(pose.plot_posture(), mimetype='image/png')


@app.route('/piecharts')
def plot_pie_charts():
    eval_posture = PostureEvaluation(os.path.join(pose.data_path, 'data.json'))
    return Response(plot_pie_charts_shoulders(eval_posture.get_percentage_shoulders()), mimetype='image/png')


def gen(pose_detector):
    while True:
        frame = pose_detector.find_landmarks()
        if pose.cap.recording:
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                pass
        else:
            break


@app.route('/video_feed')
def video_feed():
    pose.find_landmarks()
    return Response(gen(pose), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
