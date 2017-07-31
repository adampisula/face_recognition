import cv2,os
import numpy as np
from PIL import Image 
import pickle

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('recognizer/recognizer.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
path = 'dataset'

with open("users.data", "r") as f:
    users = f.readline().split(",")

cam = cv2.VideoCapture(0)
fontFace = cv2.FONT_HERSHEY_DUPLEX
fontScale = 1
fontColor = (0, 0, 0)

while True:
    ret, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
    
    for(x, y, w, h) in faces:
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(image, (x, y),(x + w, y + h), (0, 0, 0), 5)
        cv2.putText(image, users[nbr_predicted - 1], (x, y + h + 35), fontFace, fontScale, fontColor) #Draw the text
    
    cv2.imshow('Camera', image)
    cv2.waitKey(1)