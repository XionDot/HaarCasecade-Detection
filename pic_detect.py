import numpy as np
import cv2
#import keras

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
#cap = cv2.VideoCapture(0)
img = cv2.imread('image0.jpg')
    #ret, img = cap.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
faces = face_detector.detectMultiScale(gray, 1.2, 5)

   
for (x,y,w,h) in faces:
    colour = (0, 255, 0)
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #label_face = "Face" 
    cv2.putText(img, "Face", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.325, colour, 1)
    roi_gray=gray[y:(y+h), x:(x+w)]
    roi_color=img[y:(y+h), x:(x+w)]
    smile = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
    eyes = eye_detector.detectMultiScale(roi_gray, 1.1, 5)

    for (x_smile, y_smile, w_smile, h_smile) in smile: 
        cv2.rectangle(roi_color,(x_smile, y_smile),(x_smile + w_smile, y_smile + h_smile), (255, 0, 130), 1)

    for (eye_x,eye_y,eye_w,eye_h) in eyes:
        eye_centre = (x + eye_x + eye_w//2, y + eye_y + eye_h//2)
        eye_radius = round((eye_w + eye_h)*0.2)
            # Draw the circumference of the circle.             
        cv2.circle(img, eye_centre, eye_radius, (0,255,0), 1)

             # Draw a small circle (of radius 1) to show the center. 
        cv2.circle(img, eye_centre, 1, (0,0,255), 2)
        cv2.putText(img, "Eyes", tuple(np.subtract(eye_centre, (20, 25))) , cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.325, colour, 1)
cv2.imshow('frame',img)
cv2.waitKey(0)