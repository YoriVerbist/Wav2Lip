from pathlib import Path
from os import path
from glob import glob
import face_detection
import cv2, os, traceback
import numpy as np
import face_recognition
import tqdm

if not path.isfile('face_detection/detection/sfd/s3fd.pth'):
	raise FileNotFoundError('Save the s3fd model to face_detection/detection/sfd/s3fd.pth \
							before running this script!')
    
fa = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, 
                                  device='cuda')

def process_video_file(vfile):
    print(vfile)
    video_stream = cv2.VideoCapture(vfile)
    frames = []
    delete = False
    
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)
    batches = [frames[i:i + 16] for i in range(0, len(frames), 16)]

    known_face = face_recognition.load_image_file("known_face.jpeg")
    known_encoding = face_recognition.face_encodings(known_face)[0]

    i = -1
    for fb in batches:
        preds = fa.get_detections_for_batch(np.asarray(fb))

        for j, f in enumerate(preds):
            i += 1
            if f is None:
                continue
            if j % 25 == 0:
                faces = face_recognition.face_encodings(fb[j][:, :, ::-1])
                if len(faces) == 0:
                    continue
                unknown_encoding = faces[0]
                results = face_recognition.compare_faces([known_encoding], unknown_encoding)
                if not results[0]:
                    delete = True
                    break
        if delete:
            print(f'remove: {vfile}')
            os.remove(vfile)
            break


data = Path('../output/cropped')
for directory in data.iterdir():
    files = glob(path.join(directory, '*.mp4'))
    for vfile in files:
        try:
            process_video_file(vfile)
        except KeyboardInterrupt:
            exit(0)
        except :
            traceback.print_exc()
            continue
