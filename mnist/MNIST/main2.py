import cv2
from keras.models import load_model
import numpy as np

import matplotlib.pyplot as plt

model = load_model('my_model.h5')

def norm(x):
    m = np.max(x[0])
    for i in range(0, len(x[0])):
        if x[0][i]==m:
            return i
        else:
            pass

def pred(list):
    for ar in list:
        ar = np.resize(ar, (1, 784))
        x = model.predict(ar)
        print(x)
        i = norm(x)
        print(i)

im = cv2.imread('test_image.jpg')
print(im.shape[0], im.shape[1])
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

ret,thresh = cv2.threshold(imgray,121,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Detect contours using both methods on the same image
_, contours1, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

# Copy over the original image to separate variables
img1 = im.copy()
list1=[]

for c in contours1:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect

    if(w<100 and h<100 and h>40):
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # print(img1[x:x+y, y:y+h]/255)
        a = img1[y:y+h, x:x+w]/255
        cv2.imshow('a', a)
        a = cv2.resize(a, (28, 28))
        list1.append(a)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

cv2.imshow("Show", img1)
cv2.waitKey()
pred(list1)
