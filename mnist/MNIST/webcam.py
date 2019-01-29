import cv2
from keras.models import load_model
from main3 import mark

model = load_model('my_model.h5')

a = cv2.VideoCapture(0)


while True:
    ret, frame = a.read()
    cv2.imshow("lol", frame)
    if cv2.waitKey(0) & 0xff == ord('a'):
        mark(frame)
        break
a.release()
cv2.destroyAllWindows()
