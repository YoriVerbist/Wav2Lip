# app.py

import os
import sys

from flask import Flask, request, make_response, jsonify, render_template
from werkzeug.utils import secure_filename
import tqdm

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
ALLOWED_VOICE_EXTENSIONS = {'wav'} 

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_image_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def detect_faces(image):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)
    
    batch_size = 16
    while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break
    print(predictions)


@app.route('/')
def home():
    return render_template("index.html")

@app.route('/animate', methods=['POST'])
def animate():
    if 'image' not in request.files:
        return {'error': 'no image found, in request.'}, 400

    img_file = request.files['image'] 
    if img_file.filename == '':
        return {'error': 'no image found. Empty'}, 400
 
    if img_file and allowed_video_file(img_file.filename):
        #TODO just run the main script since it's just a video
        return 0


    if img_file and allowed_image_file(img_file.filename): 
        filename = secure_filename(img_file.filename)
        file.save(os.path.joing(app.config['UPLOAD_FOLDER'], filename))
        img = PILImage.create(file)
        print(img)
        # if you want a json reply, together with class probabilities:
        #return jsonify(str(pred))
        # or if you just want the result
        return {'success': img}, 200

    return {'error': 'something went wrong.'}, 500

if __name__ == '__main__':
    port = os.getenv('PORT',5000)
    app.run(debug=True, host='0.0.0.0', port=port) 
