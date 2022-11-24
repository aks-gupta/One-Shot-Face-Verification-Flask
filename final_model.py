from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread

import argparse
import _face_detection as ftk


global capture, verify
capture=0
verify = 0

# #make shots directory to save pics
# try:
#     os.mkdir('./shots')
# except OSError as error:
#     pass

#Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app
app = Flask(__name__, template_folder='./templates')

#capture video
camera = cv2.VideoCapture(0)

#face-detection module
class FaceDetection:
    verification_threshold = 0.8
    v, net = None, None
    image_size = 160

    def __init__(self):
        FaceDetection.load_models()

    @staticmethod
    def load_models():
        if not FaceDetection.net:
            FaceDetection.net = FaceDetection.load_opencv()

        if not FaceDetection.v:
            FaceDetection.v = FaceDetection.load_face_detection()

    @staticmethod
    def load_opencv():
        model_path = "./Models/OpenCV/opencv_face_detector_uint8.pb"
        model_pbtxt = "./Models/OpenCV/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(model_path, model_pbtxt)
        return net

    @staticmethod
    def load_face_detection():
        v = ftk.Verification()
        v.load_model("./Models/FaceDetection/")
        v.initial_input_output_tensors()
        return v

    @staticmethod
    def is_same(emb1, emb2):
        diff = np.subtract(emb1, emb2)
        diff = np.sum(np.square(diff))
        return diff < FaceDetection.verification_threshold, diff

    @staticmethod
    def detect_faces(image, display_images=False): # Make display_image to True to manually debug if you run into errors
        height, width, channels = image.shape

        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        FaceDetection.net.setInput(blob)
        detections = FaceDetection.net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                faces.append([x1, y1, x2 - x1, y2 - y1])

                if display_images:
                    cv2.rectangle(image, (x1, y1), (x2, y2), (174, 238, 238), 3)

        if display_images:
            print("Face co-ordinates: ", faces)
            cv2.imshow("Training Face", cv2.resize(image, (300, 300)))
            cv2.waitKey(0)
        return faces

    @staticmethod
    def load_face_embeddings(image_dir):

        embeddings = {}
        for file in os.listdir(image_dir):
            img_path = image_dir + file
            try:
                image = cv2.imread(img_path)
                faces = FaceDetection.detect_faces(image)
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    image = image[y:y + h, x:x + w]
                    embeddings[file.split(".")[0]] = FaceDetection.v.img_to_encoding(cv2.resize(image, (160, 160)), FaceDetection.image_size)
                else:
                    print(f"Found more than 1 face in \"{file}\", skipping embeddings for the image.")
            except Exception:
                print(f"Unable to read file: {file}")
        return embeddings

    @staticmethod
    def fetch_detections(image, embeddings, display_image_with_detections=False):

        faces = FaceDetection.detect_faces(image)

        detections = []
        for face in faces:
            x, y, w, h = face
            im_face = image[y:y + h, x:x + w]
            img = cv2.resize(im_face, (200, 200))
            user_embed = FaceDetection.v.img_to_encoding(cv2.resize(img, (160, 160)), FaceDetection.image_size)

            detected = {}
            for _user in embeddings:
                flag, thresh = FaceDetection.is_same(embeddings[_user], user_embed)
                if flag:
                    detected[_user] = thresh

            detected = {k: v for k, v in sorted(detected.items(), key=lambda item: item[1])}
            detected = list(detected.keys())
            if len(detected) > 0:
                detections.append(detected[0])

                if display_image_with_detections:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, detected[0], (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA, False)
                    #cv2.putText(image, "Image Matched", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        #if display_image_with_detections:
            #cv2.imshow("Detected", cv2.resize(image, (300, 300)))

        return detections


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    FaceDetection.load_models()
    embeddings = FaceDetection.load_face_embeddings("faces/")

    while True:
        success, frame = camera.read()
        if success:
            if(verify):
                #frame = cv2.flip(frame, 1)
                FaceDetection.fetch_detections(frame, embeddings, True)
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['faces', "Verified.png"])
                cv2.imwrite(p, frame)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif request.form.get('verify') == 'Verify':
            global verify
            verify = not verify

    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()
