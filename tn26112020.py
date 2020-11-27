import numpy as np
import cv2

# create a stream from the camera, the 0th video device on the system
cap = cv2.VideoCapture(0)

# these are prebuild classifiers bundled with open-cv
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# used this one to recognize the writing on the t-shirt
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

while(True):
    # grab a frame from the video feed (video is just a series of images)
    ret, frame = cap.read()

    # algorithms only work on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # detect various features
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 5)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 5)

    # draw face rectangles
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2) #blue
    
    # draw eye rectangles
    for(x,y,w,h) in eyes:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,0), 2) #cyan
    
    # draw rectangles around any text or numbers
    for(x,y,w,h) in plates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,255), 2) # yellow
    
    # draw what we have in the window
    cv2.imshow('frame', frame)

    # keep looping unless we press "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup windows and handles
cap.release()
cv2.destroyAllWindows
