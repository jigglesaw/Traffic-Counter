import cv2
import numpy as np
from time import sleep

width_min=80 
height_min=80

offset=6 

position=550 

delay= 60 

detec = []
car= 0

	
def centre_paste(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

cap = cv2.VideoCapture('video.mp4')
subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = subtractor.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_view = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilated_view = cv2.morphologyEx (dilated_view, cv2. MORPH_CLOSE , kernel)
    contours,h=cv2.findContours(dilated_view,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (25, position), (1200, position), (255,127,0), 3) 
    for(i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        valid_contours = (w >= width_min) and (h >= height_min)
        if not valid_contours:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        center = centre_paste(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(position+offset) and y>(position-offset):
                car+=1
                cv2.line(frame1, (25, position), (1200, position), (0,127,255), 3)  
                detec.remove((x,y))
                print("car is detected : "+str(car))        
       
    cv2.putText(frame1, "VEHICLE COUNT : "+str(car), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detected",dilated_view)

    if cv2.waitKey(1) == 27:
        break
    
cv2.destroyAllWindows()
cap.release()
