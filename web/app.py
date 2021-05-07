# app.py

import os
import sys

from flask import Flask, request, make_response, jsonify, render_template
from werkzeug.utils import secure_filename
import tqdm, torch, cv2, ast, pathlib
import numpy as np
import pandas as pd

# Import face_detection library
sys.path.append("../")
import face_detection

# Folder where files are stored
UPLOAD_FOLDER = "static/uploads"

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'} 
ALLOWED_VIDEO_EXTENSIONS = {'mp4'} 
ALLOWED_AUDIO_EXTENSIONS = {'wav'} 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global image_path
global audio_path
global cache_file

command = 'python3 ../inference.py --checkpoint_path ../checkpoints/wav2lip_gan.pth --face {} --audio {} \
                                   --outfile {} --person {} --static_video {} --cache {}'

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
            predictions.extend(detector.get_detections_for_batch(np.array(image), False))
            break
        except:
            pass
    del detector
    return predictions


def detect_faces_batch(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)
    
    batch_size = 32
    while 1:
        predictions = []
        try:
            for i in range(0, len(images), batch_size):
        	    predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size]), True))
        except RuntimeError:
            if batch_size == 1: 
        	    raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break
    del detector
    return predictions


def filename_exists(path):
    file = path.split('/')[-1]
    path = '/'.join(path.split('/')[0:2])
    files = os.listdir(path)
    return file in files


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    global image_path, audio_path, cache_file

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
    
    filename = image_path.split('/')[-1].split('.')[0]
    filename = 'static/results/' + filename + '.mp4'

    cache_file = f"static/cache/{image_filename.split('.')[0] + '.csv'}"

    # If the face detections are already cached, just use those instead of detecting them again
    if filename_exists(cache_file):
        #TODO if the cache file exists, just get the detections from there
        df = pd.read_csv(cache_file)
        preds = df.values.tolist()
        if len(preds) == 1:
            preds = [[eval(pred) for pred in preds[0]]]
        elif len(preds) > 1:
            tmp = []
            for batch in preds:
                new_batch = []
                for pred in batch:
                    new_batch.append(eval(pred))
                tmp.append(new_batch)
            preds = tmp
            pass
        elif len(preds[0]) < 1:
            return {'error': 'No faces found in the picture/video'}, 400
        

    # If the face detections aren't cached, predict them and save them to a csv file
    if not filename_exists(cache_file):
        # Get the face detections of the first frame of the video
        if img_file and allowed_video_file(img_file.filename) and request.form.get('static_video'):
            video_stream = cv2.VideoCapture(img_file)
            success, image = video_stream.read()
            video_stream.release()
            preds = detect_faces(image.copy())

        # Get the face detections of the whole video
        if img_file and allowed_video_file(img_file.filename) and not request.form.get('static_video'):
            video_stream = cv2.VideoCapture(f"static/uploads/{img_file.filename}")
            full_frames = []
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break

                full_frames.append(frame)
            preds = detect_faces_batch(full_frames.copy())
            print(preds)
        
        # Get the face detections of the image
        if img_file and allowed_image_file(img_file.filename):
            image = [cv2.imread(f"static/uploads/{image_filename}")]
            preds = detect_faces(image.copy())

        df = pd.DataFrame(preds)
        print(f"Saved the predictions to {cache_file}")
        df.to_csv(cache_file, index=False)

    print(request.form.get('static_video'))
    print('preds:', preds)
    # If it's not a static video, just run the script on the video
    if img_file and allowed_video_file(img_file.filename) and not request.form.get('static_video'):
        print('non static video')
        os.system(command.format(image_path, audio_path, '{}'.format(filename), 0, 0, cache_file))
        return render_template("animated.html", filename = filename.strip('static/'))

    # If it's a static video, select the correct face first
    if img_file and allowed_video_file(img_file.filename) and request.form.get('static_video'):
        if len(preds[0]) == 1:
            print('static video')
            os.system(command.format(image_path, audio_path, '{}'.format(filename), 0, 1, cache_file))
            return render_template("animated.html", filename = filename.strip('static/'))

        return render_template("select_face.html", preds = preds[0], filename = 'uploads/' + image_filename.split('.')[0] + '.jpg')
    
    # If only one face is detected run the script without selecting the correct face
    if img_file and allowed_image_file(img_file.filename) and len(preds[0]) == 1:
        os.system(command.format(image_path, audio_path, '{}'.format(filename), 0, 1, cache_file))
        return render_template("animated.html", filename = filename.strip('static/'))

    # If it's a image with multiple faces, select the correct face first
    if img_file and allowed_image_file(img_file.filename):
        return render_template("select_face.html", preds = preds[0], filename = 'uploads/' + image_filename)

    return {'error': 'something went wrong.'}, 500


@app.route('/select_face', methods=['POST'])
def select_face():
    global image_path, audio_path, cache_file

    filename = image_path.split('/')[-1].split('.')[0]
    filename = 'static/results/' + filename + '.mp4'
    print(filename)

    face = int(request.form['faces'])

    os.system(command.format(image_path, audio_path, '{}'.format(filename), face, 1, cache_file))
    return render_template("animated.html", filename = filename.strip('static/'))


if __name__ == '__main__':
    port = os.getenv('PORT',5000)
    app.run(debug=True, host='0.0.0.0', port=port) 
