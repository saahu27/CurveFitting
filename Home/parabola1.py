import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import imutils

x1 = []
y1 = []
# x2 = []
# y2 = []

capture = cv.VideoCapture(r'C:\Users\sahru\OneDrive\Desktop\ENPM673 - code\Getting started - ENPM673 Spring 2021\ball_video1.mp4')

if(capture.isOpened() == False):
    print ("!!! Failed VideoCapture: unable to open file!")

    # red_channel = frame[:,:,1]

while True:
    isTrue,frame = capture.read()
    if isTrue:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

# Find contours and extract the bounding rectangle coordintes
# then find moments to obtain the centroid
        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
    # Obtain bounding box coordinates and draw rectangle
            print(c)
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x + w, y + h), (36,255,12), 2)

    # Find center coordinate and draw center point
            M = cv.moments(c)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv.circle(frame, (cx, cy), 2, (36,255,12), -1)
            print('Center: ({}, {})'.format(cx,cy))

            # x1.append(cx + h)
            # y1.append(cy + h)
            # x2.append(cx - h)
            # y2.append(cy - h)
            x1.append(cx)
            y1.append(len(gray) - cy)

        cv.imshow('video',frame)
    else:
        break
    
    if cv.waitKey(100) & 0xFF==ord('d'):
        break


capture.release()
cv.destroyAllWindows()
# x1.extend(x2)
# y1.extend(y2)
y1 = np.row_stack(y1)

x_sq = np.power(x1, 2)

    ## A = [x^2  x  1]
A = np.column_stack((x_sq, x1, np.ones((len(x1)))))
X_dagger = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)
weights = np.dot(X_dagger, y1)
print(weights)

#ax^2 + bx + c = y
line = A.dot(weights)
plt.plot(x1, y1, 'o')
plt.plot(x1,line)
plt.show()
