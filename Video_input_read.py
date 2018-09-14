#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 11:23:03 2018

@author: prayash
"""

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import numpy as np
import cv2

cv2.__version__
help(cv2.ml)

#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

cap = cv2.VideoCapture('/Users/prayash/Downloads/Bowler_clip.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
 
    #net.setInput(blob)
    #detections = net.forward()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()