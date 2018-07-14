#!/usr/bin/env python
#-*-coding:utf-8-*-
from flask import Flask
import time
from flask import render_template
from flask import request
import base64
from PIL import Image
from io import StringIO, BytesIO
import robo_talker
import cv2
import numpy as np
import rospy
from multiprocessing import Pipe, Process
from sensor_msgs.msg import Image
import face_recognition

pipe = Pipe()

def draw_location(img_path, message):
    unknown_image = cv2.imread(img_path)
    small_image = cv2.resize(unknown_image, (0, 0), fx=0.25, fy=0.25)
    rgb_pic = small_image[:, :, ::-1]
    face_location = face_recognition.face_locations(rgb_pic)
    if len(face_location) == 0:
        return "can't find a face"
    face_location = face_location[0]
    # face_encoding = face_recognition.face_encodings(rgb_pic, face_location)[0]
    (top, right, bottom, left) = face_location
    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    # Draw a box around the face
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(unknown_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(unknown_image, message, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imwrite(img_path, unknown_image)
    # cv2.imshow('Video', unknown_image)

def extract_face_features(gray_face, wd=96, ht=96):
    gray_face = cv2.resize(gray_face, (wd, ht))
    gray_face = cv2.equalizeHist(gray_face)
    gray_face = cv2.GaussianBlur(gray_face, ksize=(3, 3),
                                    sigmaX=0, sigmaY=0)

    gray_face = np.array(gray_face, dtype=np.float64)
    gray_face_vector = gray_face.reshape(wd * ht, order='C')
    return gray_face_vector

app = Flask(__name__)

@app.route('/')
def webcam():
    return render_template('index.html')

@app.route('/upload', methods=['POST','GET'])
def upload():
    print('getting data from web.')
    if request.method == 'POST':
        image_b64 = request.form['img']
        imgdata = base64.b64decode(image_b64)
        with open('org_img.jpg', 'wb') as f:
            f.write(imgdata)
            f.close()
        imgdata = cv2.imread('org_img.jpg')
        cv2.imwrite('org_img.jpg',imgdata)
        image = face_recognition.load_image_file('org_img.jpg')
        en_array = face_recognition.face_encodings(image)
        if len(en_array) == 0:
            return "result: can't find face"
        en_array = en_array[0]
        pipe[1].send(en_array)
        imgdata = cv2.cvtColor(imgdata, cv2.COLOR_BGR2GRAY) 
        face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(
                                            imgdata,
                                            scaleFactor=1.15,
                                            minNeighbors=5,
                                            minSize=(5, 5)  # ,
                                                    # flags=cv2.CV_HAAR_SCALE_IMAGE
                                                        )
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(imgdata, (x, y), (x + w, y + w), (255, 0, 0), 2)
            gray_face = imgdata[y:y+h, x:x+w]
            gray_face_vector = extract_face_features(gray_face)
            gray_face = gray_face_vector.reshape((96, 96))
            cv2.imwrite('./gray_face.jpg', gray_face)
			

        #time.sleep(3)
        result = pipe[1].recv()
        #if result != 'unKnown' || result != "can't find a face":
         #   draw_location(img_path,message) 
        return "result:"+result
        #with open('pic.jpg', 'wb') as f:
        #    f.write(imgdata)
    #return render_template('index.html')
if __name__ == '__main__':
    
    
    p1=Process(target=robo_talker.talker,args=(pipe[0],))
    p1.start()
        #thread.start_new_thread(app.run,(host='0.0.0.0', port=7777, ssl_context='adhoc'))
    
    app.run(host='0.0.0.0', port=7777, ssl_context='adhoc')
    p1.join()
    
