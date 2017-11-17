import cv2
import sys

face_cascade_path = "/usr/local/Cellar/opencv/2.4.7.1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
smile_cascade_path = "/usr/local/Cellar/opencv/2.4.7.1/share/OpenCV/haarcascades/haarcascade_smile.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

sF = 1.05

while True:
    _, frame = cap.read()

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor= sF,
        minNeighbors=8,
        minSize=(55, 55),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE #required by cv2
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        smile = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE #required by cv2
            )

        for (x, y, w, h) in smile:
            print "Found", len(smile), "smiles!"
            cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 1)

    cv2.imshow('Smile Detector', frame)
    c = cv2.cv.WaitKey(7) % 0x100
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
