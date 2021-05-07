import numpy as np
import cv2 as cv


def rescaleFrame(frame, scale=0.75):
    width = (int)(frame.shape[1] * scale)
    height = (int)(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

haar_cascade = cv.CascadeClassifier("haar_face.xml")

people = ['Me', 'Nicholas Cage']
#features = np.load('features.npy')
#labels = np.load('labels.npy')

print(dir (cv.face))

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.read('face_trained.yml')

#img = cv.imread("Photos/Nicholas Cage/NC2.jpg")
img = rescaleFrame(cv.imread("Photos/Nicholas Cage/NC12.jpg"), .9)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect Face

faces_rect = haar_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)

    print(f'Label = {people[label]} with a confidence of {confidence}')
    
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    
    color = (0, 0, 0)

    if label == 0:
        color = (0,0,255)
    else:
        color = (255,0,0)

    cv.rectangle(img, (x,y), (x+w,y+h), color, thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)
