import cv2
import os

path = os.path.dirname(os.path.abspath(__file__))
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(path + r"/trainer/trainer.yml")
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)

    for x, y, w, h in faces:
        nbr_predicted, conf = recognizer.predict(gray[y: y + h, x: x + w])

        if conf < 50:  # Установите порог уверенности
            if nbr_predicted == 1:
                nbr_predicted = "Albert"
        else:
            nbr_predicted = "Unknown"

        cv2.rectangle(im, (x - 50, y - 50), (x + w + 50, y + h + 50), (225, 0, 0), 2)
        cv2.putText(im, str(nbr_predicted), (x, y + h), font, 1.1, (0, 255, 0))
        cv2.imshow("Face recognition", im)
        cv2.waitKey(10)
