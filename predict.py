import cv2
import glob
from keras.models import load_model
import numpy as np
import time
import tensorflow as tf
import serial
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))



model = load_model("E:\final\model_best_weights.h5")
# model.summary()
cap = cv2.VideoCapture(0)

#sc=serial.Serial('COM3',115200,timeout=.1)
pos=90
while(cap.isOpened()):
    
#while 1:

    time.sleep(3)
    ret,frame=cap.read()
    
    img=frame
    
    

 # imgPath = input("input image: ")
  #if imgPath == "":
    #break

    #img = cv2.imread(imgPath)
    img = cv2.resize(img, (32,32))
    img = img.astype(np.float32)
    img = img/255
  # print(img)

    img = img[np.newaxis, ...]

    start_time = time.time()
    result = model.predict(img)
    result = np.squeeze(result)
    print(result)
    result = (round(result[0], 2),round(result[1], 2),round(result[2], 2))
    print("--- %s seconds ---" % (time.time() - start_time))
    print(result)
    
    
    
    
    
    if(result[0]>=result[1]):
        if result[0]>=result[2]:
            print("grade : A")
            grade='A'
            pos=40
        
    if result[1]>=result[0]:
        if result[1]>=result[2]:
            print("grade :B")
            grade='B'
            pos=90
        else:
            print("grade :C")
            grade='C'
            pos=150
    if result[2]>=result[0]:
        if result[2]>=result[1]:
            print("grade :C")
            grade='C'
            pos=150    
    #sc.write(bytes(grade,'utf-8'))
    #time.sleep(0.5)
    #print(str(sc.readline()))

cap.release()
cv2.destroyAllWindows()

