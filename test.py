import cv2
import numpy as np
import pandas as pd
from scipy.special import softmax

def relu(x):
    x[x<0]=0
    return(x)

while(1):
    drawing = False # true if mouse is pressed
    ix,iy = -1,-1

    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        global ix,iy,drawing,mode

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.circle(img,(x,y),1,255,-1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.circle(img,(x,y),1,255,-1)
    img = np.zeros((28,28), np.uint8)
    cv2.namedWindow('Enter to predict, esc to exit')
    cv2.setMouseCallback('Enter to predict, esc to exit',draw_circle)

    cont = True
    while(1):
        cv2.imshow('Enter to predict, esc to exit',img)
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            break
        if k == 27:
            cont = False
            break
    cv2.destroyAllWindows()
    img1 = (img / 255).astype(np.float32).reshape((1,784))
    w1 = np.array(pd.read_csv('w1.csv').iloc[:,1:]).astype(np.float32)
    b1 = np.array(pd.read_csv('b1.csv').iloc[:,1:]).astype(np.float32).reshape((300,))
    w2 = np.array(pd.read_csv('w2.csv').iloc[:,1:]).astype(np.float32)
    b2 = np.array(pd.read_csv('b2.csv').iloc[:,1:]).astype(np.float32).reshape((200,))
    w3 = np.array(pd.read_csv('w3.csv').iloc[:,1:]).astype(np.float32)
    b3 = np.array(pd.read_csv('b3.csv').iloc[:,1:]).astype(np.float32).reshape((10,))   
    Yhat = softmax((np.dot(relu((np.dot(relu((np.dot(img1,w1) + b1)),w2) + b2)),w3) + b3),axis=1)
    if not img1.sum() < 0.0001:
        print(np.argmax(Yhat),end='')
    if not cont:
        break
    else:
        img = np.zeros((28,28), np.uint8)

print('')

