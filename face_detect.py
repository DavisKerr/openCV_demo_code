import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    width = (int)(frame.shape[1] * scale)
    height = (int)(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = rescaleFrame(cv.imread('Photos/people.jpg'), 0.5)
#img = rescaleFrame(cv.imread('Photos/fam.jpg'), 0.1)
cv.imshow("Me", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Me Gray', gray)

haar_cascade = cv.CascadeClassifier("haar_face.xml")

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.07, minNeighbors=1)

print(f'Number of faces found {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

cv.imshow("Faces", img)

cv.waitKey(0)