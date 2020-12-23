import numpy as np
import cv2
#import keras
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
cap = cv2.VideoCapture(0)
#img = cv2.imread('image0.jpg')
i = 0
while 1:
    
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        # center = (x + w//2, y + h//2)
        # cv2.ellipse(img, center, (w//2, h//2), 0, 0, 360, (153, 0, 255), 2)
        i += 1
        cv2.rectangle(img, (x + 10, y - 20), (x+w, y+h),(0, 255, 0), 1)
        cv2.putText(img, "Face", (x + 10, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.325, (0, 255, 0), 1)
        #print("Face DETECTED")
        roi_gray=gray[y:(y+h), x:(x+w)]
        roi_color=img[y:(y+h), x:(x+w)]
        smile = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        eyes = eye_detector.detectMultiScale(roi_gray, 1.1, 5)

        for (x_smile, y_smile, w_smile, h_smile) in smile: 
            center = (x + x_smile + w_smile//2, y + y_smile + h_smile//2)
            cv2.ellipse(img, center, (w_smile//2, h_smile//2), 0, 0, 360, (0, 255, 0), 1)

            #cv2.rectangle(roi_color,(x_smile, y_smile),(x_smile + w_smile, y_smile + h_smile), (0, 255, 0), 2)


        for (eye_x,eye_y,eye_w,eye_h) in eyes:
            eye_centre = (x + eye_x + eye_w//2, y + eye_y + eye_h//2)
            eye_radius = round((eye_w + eye_h)*0.2)
            # Draw the circumference of the circle.             
            cv2.circle(img, eye_centre, eye_radius, (0,255,0), 1)

             # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(img, eye_centre, 1, (0,0,255), 1)
            cv2.putText(img, "Eyes", tuple(np.subtract(eye_centre, (20, 25))) , cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.325, (0, 255, 0), 1)
            #print("Eyes DETECTED")
        
    
    print(i)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()