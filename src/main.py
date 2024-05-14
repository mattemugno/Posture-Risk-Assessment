import os
from flask import Flask, Response, render_template, request
from src.PoseDetector import PoseDetector
from src.PostureEvaluation import PostureEvaluation

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
template_folder = os.path.join(base_dir, 'template')

app = Flask(__name__, template_folder=template_folder)


class PoseManager:
    def __init__(self):
        self.pose = None
        self.recording = False

    def create_pose_detector(self):
        if self.pose is None:
            self.pose = PoseDetector(model_complexity=1, enable_segmentation=False)

    def destroy_pose_detector(self):
        if self.pose is not None:
            del self.pose
            self.pose = None


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
        elif request.form.get('stop') == 'Stop/Start':
            if pose_manager.recording:
                pose_manager.destroy_pose_detector()
                pose_manager.recording = False
            else:
                pose_manager.create_pose_detector()
                pose_manager.recording = True
        elif request.form.get('save') == 'Save Data':
            pose_manager.pose.save_data()
    elif request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')


@app.route('/plot')
def plot_posture_in_time():
    if pose_manager.pose is None:
        return "Pose detector not initialized"
    eval_posture = PostureEvaluation(os.path.join(pose_manager.pose.data_path, 'data.json'))
    return Response(eval_posture.plot_posture(), mimetype='image/png')


@app.route('/piecharts')
def plot_pie_charts():
    if pose_manager.pose is None:
        return "Pose detector not initialized"
    eval_posture = PostureEvaluation(os.path.join(pose_manager.pose.data_path, 'data.json'))
    eval_posture.calculate_percentages()
    return Response(eval_posture.plot_pie_charts(), mimetype='image/png')


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
