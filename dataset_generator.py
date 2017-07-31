import numpy as np
import cv2

detector = cv2.CascadeClassifier('Classifiers/face.xml')
i = 0

with open("users.data", "r") as f:
    users = f.readline().split(",")

name = raw_input("Your name: ")

with open("users.data", "a") as f:
    f.write("," + name)

#id = len(users)
id = 0

cam = cv2.VideoCapture(0)

while True:
    ret, image = cam.read()

    cv2.imshow("Camera", image)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            i += 1

            cv2.imshow("Face", gray[y:y + h, x:x + w])
            cv2.imwrite("dataset/face-" + str(id) + '.' + str(i) + ".jpg", gray[y:y + h, x:x + w])

            print "Took photo #" + str(i)

    print "\t" + str(i)

cam.release()
cv2.destroyAllWindows()