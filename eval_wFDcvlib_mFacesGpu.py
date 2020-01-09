## eval_wFDmultiFaces.py: This code will perform testing on images, where face detection module is considered
## It will yield confusion matrices for age model and gender model
## Age model => Compact ResNet54
## Gender model => RoR

import os
##os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
##os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use id from $ nvidia-smi

import random
import math
from pathlib import Path
from scipy.io import loadmat
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import sys
import cvlib as cv
from keras.utils import Sequence, to_categorical
from datetime import datetime
from tqdm import tqdm
import argparse
from contextlib import contextmanager
from keras.utils.data_utils import get_file
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.contrib.memory_stats.python.ops import memory_stats_ops
from keras import backend as K
import time
from sklearn.metrics import f1_score, mean_absolute_error
from keras.utils import multi_gpu_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from keras.models import Model, Sequential, load_model
import traceback

def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                                                 "and estimates age for the detected faces.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vmargin", type=float, default = 0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--hmargin", type=float, default = 0.4,
                        help="margin around detected face for age-gender estimation")
    parser.add_argument("--results_dir", type=str, default = "results",
                        help="Results directory to store results of predictions")
    args = parser.parse_args()
    return args
        
def main():
    
#    num_cores = 4
#    num_CPU = 1
#    num_GPU = 0

#    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
#                            inter_op_parallelism_threads=num_cores, 
#                            allow_soft_placement=True,
#                            device_count = {'CPU' : num_CPU,
#                                            'GPU' : num_GPU}
#                           )

#    session = tf.Session(config=config)
#    K.set_session(session)
    
    args = get_args()
    model_name = args.model_name

    # The directory containing list of directories of images
    rootpath = "./multiple faces 884"
    
    vmargin = args.vmargin
    hmargin = args.hmargin
    results_dir = args.results_dir

    # for face detection
    #detector = dlib.get_frontal_face_detector()

    gweights= "./models/ResNet50_FaceNetWt_Gen_final_model.hdf5"
    #print("Gender model loading")
    gender_model = load_model(gweights)
    #print("End Gender model loading")
    gimg_size = 160 #model.input.shape.as_list()[1]

    aweights = "./models/ResNet54_ImageNet_Compact_final_model.hdf5"
    age_model = load_model(aweights)
    aimg_size = 224
    print("Demogrphics model loaded successfully.")
    fdend = 0
    ageend = 0
    fdtime = 0
    agetime = 0
    gentime = 0
    count = 1
    nondetectedfaces = 0
    nfaces = 0

    notfiles = [
        "gaurav_kravi_other_39_0_6.jpg",
        "prernasangeetakravi_28_1_2.jpg",
        "prernasangeetakravi_28_1_6.jpg"]
                
    res = list()
    startflag = False
    temppath = ""

    try:
        dirs = os.listdir(rootpath)
        for dir1 in dirs:
##            if dir1 == "prerna-sangeeta-kravi_3_30_28_37_1_1_0":
##                startflag = True

##            if startflag == False:
##                continue
                
            age1 = list()
            gender1 = list()
            innerdir = rootpath + "/" + dir1
            files = os.listdir(innerdir)
            arr = dir1.split("_")
            #print(arr)
            nfaces = int(arr[1])

            if nfaces == 2:
                age1.append(int(arr[2]))
                gender1.append(int(arr[4]))
                age1.append(int(arr[3]))
                gender1.append(int(arr[5]))
            elif nfaces == 3:
                age1.append(int(arr[2]))
                gender1.append(int(arr[5]))
                age1.append(int(arr[3]))
                gender1.append(int(arr[6]))
                age1.append(int(arr[4]))
                gender1.append(int(arr[7]))


            for file1 in files:
##                print(count)
##                count += 1
##                if count > 5:
##                    break
##                temp1 = 0
                
                if not file1.endswith(".jpg"):
                    continue
                if file1 in notfiles:
                    continue
                
                temppath = str(innerdir + "/" + file1)
                #print(temppath)
                img = cv2.imread(temppath, 1)

                if img is not None:
                    #input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_h, img_w, _ = np.shape(img)

                    starttime = time.time()
##                    # detect faces using dlib detector
##                    detected = detector(input_img, 1)

                    #print("Face detection ...")
                    # detect faces using cvlib
                    face, confidence = cv.detect_face(img)
                    fdend = time.time()
                    #print("End Face detection ...")
                    fdtime = fdtime + fdend - starttime
                    #Sort faces from left to right respective to x axis
                    face.sort(key=lambda k: k[0])

                    afaces = np.empty((len(face), aimg_size, aimg_size, 3))
                    gfaces = np.empty((len(face), gimg_size, gimg_size, 3))
                    
                    if len(face) > 0:
                        for j, f in enumerate(face):
                            (x1, y1) = f[0], f[1]
                            (x2, y2) = f[2], f[3]
                            w = x2-x1
                            h = y2-y1

                            xw1 = max(int(x1 - hmargin * w), 0)
                            yw1 = max(int(y1 - vmargin * h), 0)
                            xw2 = min(int(x2 + hmargin * w), img_w - 1)
                            yw2 = min(int(y2 + vmargin * h), img_h - 1)

                            afaces[j, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (aimg_size, aimg_size))
                            gfaces[j, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (gimg_size, gimg_size))

                    agestart = time.time()
                    # predict ages and genders of the detected faces
                    pred_ages = age_model.predict(afaces)
                    ageend = time.time()
                    agetime = agetime + ageend - agestart
                    results = gender_model.predict(gfaces)
                    gentime = gentime + time.time() - ageend
                    if (len(pred_ages) == 0) or (len(results) == 0):
                        pass

                    else:    
                        for idx, item in enumerate(pred_ages):
                            pgend = 1 if results[idx][0] <= results[idx][1] else 0
                            res.append([temppath, age1[idx], int(round(item[0], 0)), gender1[idx], pgend]) 
    finally:
        #print(temppath)
        df1 = pd.DataFrame(res, columns=["FileName", "Age", "Pred_Age", "Gender", "Pred_Gender"]) #, "GenProb", "FaceXpos"])
        df1.to_csv("./results_dir/prob_results.csv", index = False)
        ## Dropping non-predicted age
        actage = list()
        for _, age in df1['Age'].iteritems():
            if age < 18:
                actage.append("Centenials")
            elif age <= 35:
                actage.append("Millenials")
            elif age <= 54:
                actage.append("GenX")
            elif age > 54:
                actage.append("BabyBoomers")

        predage = list()
        for _, age in df1['Pred_Age'].iteritems():
            if age < 18:
                predage.append("Centenials")
            elif age <= 35:
                predage.append("Millenials")
            elif age <= 54:
                predage.append("GenX")
            elif age > 54:
                predage.append("BabyBoomers")

        df2 = pd.DataFrame({"Age" : actage, "Pred_Age" : predage})
        cf = confusion_matrix(df2["Age"], df2["Pred_Age"]) 
        cr = classification_report(df2["Age"], df2["Pred_Age"]) 
        mae = mean_absolute_error(df1["Age"], df1["Pred_Age"])
        
        print("Age model results ===> ")
        print(aweights)
        print(cf)
        print(cr)
        
        actgender = list()
        for _, gender in df1['Gender'].iteritems():
            if gender == 1:
                actgender.append("Female")
            else:
                actgender.append("Male")

        predgender = list()
        for _, gender in df1['Pred_Gender'].iteritems():
            if gender == 1:
                predgender.append("Female")
            else:
                predgender.append("Male")

        df2 = pd.DataFrame({"Gender" : actgender, "Pred_Gender" : predgender})
        cf = confusion_matrix(df2["Gender"], df2["Pred_Gender"])
        cr = classification_report(df2["Gender"], df2["Pred_Gender"]) 
        mae = mean_absolute_error(df1["Gender"], df1["Pred_Gender"])
        
        print("Gender model results ===>")
        print(gweights)
        print(cf)
        print(cr)
        
    print("Face detection time %s" %fdtime)
    print("Age estimation time %s" %agetime)
    print("Gender prediction time %s" %gentime)
        
def main1():
    t1 = TestGenerator()

if __name__ == '__main__':
    main()    
