# app.py

import os
import sys

from flask import Flask, request, make_response, jsonify, render_template
from werkzeug.utils import secure_filename
import tqdm, torch, cv2
import numpy as np

# Import face_detection library
sys.path.append("../")
import face_detection

# Folder where files are stored
UPLOAD_FOLDER = "static/uploads"

# codeblock below is needed for Windows path #############
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
##########################################################

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'} 
ALLOWED_VIDEO_EXTENSIONS = {'mp4'} 
ALLOWED_AUDIO_EXTENSIONS = {'wav'} 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global image_path
global audio_path
global box

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

def allowed_audio_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def detect_faces(image):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)
    
    predictions = []
    while 1:
        try:
            predictions.extend(detector.get_detections_for_batch(np.array(image)))
            break
        except:
            pass
    return predictions


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    global image_path, audio_path, box
    if 'image' not in request.files or 'audio' not in request.files:
        return {'error': 'no image or audio found, in request.'}, 400

    img_file = request.files['image'] 
    audio_file = request.files['audio']
    if img_file.filename == '' or audio_file.filename == '':
        return {'error': 'no image or audio file found. Empty'}, 400
    
    if not allowed_audio_file(audio_file.filename):
        return {'error': 'Audio file needs to be WAV format.'}, 400
    
    image_filename = secure_filename(img_file.filename)
    audio_filename = secure_filename(audio_file.filename)
    img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
    audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], audio_filename))
    image_path = 'static/uploads/' + image_filename
    audio_path = 'static/uploads/' + audio_filename
    
    if img_file and allowed_video_file(img_file.filename):
        #TODO just run the main script since it's just a video
        
        return {'success': img_file.filename}, 200


    if img_file and allowed_image_file(img_file.filename): 

        image = [cv2.imread(f"static/uploads/{image_filename}")]

        preds = detect_faces(image.copy())
        print(preds)

        box = preds

        if len(preds) == 1:
            #TODO just run the normal script and skip the select face page
            box = list(box[0])
            os.system('python3 ../inference.py --checkpoint_path ../checkpoints/wav2lip_gan.pth --face {} --audio {} --outfile {} --box {}'.format(image_path, audio_path, 'static/results/result_voice.mp4', box))

        return render_template("select_face.html", preds = preds, filename = 'uploads/' + image_filename)

    return {'error': 'something went wrong.'}, 500

@app.route('/select_face', methods=['POST'])
def select_face():
    global image_path, audio_path, box
    if request.method == 'POST':
        face = int(request.form['faces'])

        box = str(box[face]).strip('()')
        print(box)

        os.system('python3 ../inference.py --checkpoint_path ../checkpoints/wav2lip_gan.pth --face {} --audio {} --outfile {} --box {}'.format(image_path, audio_path, 'static/results/result_voice.mp4', box))
        return render_template("animated.html")

if __name__ == '__main__':
    port = os.getenv('PORT',5000)
    app.run(debug=True, host='0.0.0.0', port=port) 
