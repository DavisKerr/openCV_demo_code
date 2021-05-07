import cv2 as cv

haar_cascade_upper = cv.CascadeClassifier("haar_upper.xml")

capture = cv.VideoCapture("output.avi")

while True:
    isTrue, frame = capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    upperbody_rect = haar_cascade_upper.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=2)

    cv.circle(frame, (300,450), 10, (0,0,255), thickness=-1)

    for (x,y,w,h) in upperbody_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv.line(frame, (x+(w//2), y+(h//2)), (300, 450), (255,0,0), 2)
    cv.imshow("Video", frame)
    if cv.waitKey(5) & 0xFF == ord('d'):
        break


'''while True:
    isTrue, frame = capture.read()
    cv.imshow("Video", frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break'''

capture.release()

cv.destroyAllWindows()



