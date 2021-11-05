import cv2 
import os
import h5py
import numpy as np
from math import atan2, pi

# This method get a Cascade classifier, an image
# and eyes position and extract face from image
# and rotate and cropp it.
def preprocess(clf,img,eyes,padding=0):
    g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces_detected = clf.detectMultiScale(g)
    x, y, w, h = faces_detected[0]
    rows, cols = img.shape[:2]
    xd = int(eyes[3]) - int(eyes[1])
    yd = int(eyes[2]) - int(eyes[0])
    angel = atan2(yd,xd)*180/pi
    x_center = (int(eyes[3]) + int(eyes[1]))/2
    y_center = (int(eyes[2]) + int(eyes[0]))/2
    M = cv2.getRotationMatrix2D((y_center,x_center),angel,1)
    img_rotated = cv2.warpAffine(img, M, (cols,rows))
    croped_image = img_rotated[y-padding:y+h+padding , x-padding:x+w+padding]
    return cv2.resize(croped_image,dsize=(224,224))

# This method get a dataset and use fisher_score
# to rate features.
def fisher_score(dataset):
    total_mean = dataset[:,1:].mean(axis=0)
    total_std = dataset[:,1:].std(axis=0)**2

    positive_mean = dataset[dataset[:,0] == 1][:,1:].mean(axis=0)
    positive_std = dataset[dataset[:,0] == 1][:,1:].std(axis=0)**2

    negative_mean = dataset[dataset[:,0] == 0][:,1:].mean(axis=0)
    negative_std = dataset[dataset[:,0] == 0][:,1:].std(axis=0)**2

    n_positive = dataset[dataset[:,0] == 1].shape[0]
    n_negative = dataset[dataset[:,0] == 1].shape[0]

    positive_numerator = n_positive * (positive_mean - total_mean)**2
    negative_numerator = n_negative * (negative_mean - total_mean)**2

    positive_denominator = n_positive * positive_std**2
    negative_denominator = n_negative * negative_std**2

    numerator = positive_numerator + negative_numerator
    denominator = positive_denominator + negative_denominator

    ranks = numerator / denominator

    return ranks