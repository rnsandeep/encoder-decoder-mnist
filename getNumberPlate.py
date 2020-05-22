import os, shutil, sys, cv2
import os
from xml.dom import minidom
os.environ['GLOG_minloglevel'] = '2'
import  os, sys, cv2
import argparse
import shutil
import sys ,os
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np
import pickle

def files_with_ext(mypath, ext):
    files = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and os.path.splitext(os.path.join(mypath, f))[1] == ext]
    return files



xmls = files_with_ext(sys.argv[1], '.xml')
images = files_with_ext(sys.argv[2], '.jpg')

def read_xml(xml_path):
    flag = True
    doc = minidom.parse(xml_path)
    objects = doc.getElementsByTagName("object")
    fname = doc.getElementsByTagName("filename")[0].firstChild.data
    size = doc.getElementsByTagName("size")[0]
    width = size.getElementsByTagName("width")[0].firstChild.data
    height =  size.getElementsByTagName("height")[0].firstChild.data
    labelledObjects = dict()

    for obj in objects:
        name = obj.getElementsByTagName("name")[0].firstChild.data
        xmin = obj.getElementsByTagName("xmin")[0].firstChild.data
        ymin = obj.getElementsByTagName("ymin")[0].firstChild.data
        xmax = obj.getElementsByTagName("xmax")[0].firstChild.data
        ymax = obj.getElementsByTagName("ymax")[0].firstChild.data
       
        
        if int(ymin) >= int(height) or int(ymax) >= int(height) or int(xmin) >= int(xmax) or int(ymin) >= int(ymax) or int(xmin) >= int(width) or int(xmax) >= int(width) :
              flag = False
              continue
        if int(xmin) <=0 or int(ymin) <=0 or int(xmax) <=0 or int(ymax) <=0:
              flag = False
              continue

        if (int(xmax)- int(xmin))*(int(ymax)-int(ymin)) <= 0: 
              flag = False
              continue
   
        box = [xmin, ymin, xmax, ymax]
        bb = []
        for b in box:
            if int(b)  ==0:
                b = 1
            bb.append(b)
        box = (bb[0], bb[1], bb[2] , bb[3])
        if name not in labelledObjects:
            labelledObjects[name] = []
            labelledObjects[name].append(box)
        else:
            labelledObjects[name].append(box)

    if len(labelledObjects.keys())== 0:
        return False, xml_path

    return labelledObjects


from statistics import mean
import numpy as np


from sklearn.cluster import KMeans
import numpy as np

def getSortedBoxes(boxes, names):
    
    mid_points = [(1, ((box[1]+box[3])/2)) for box in boxes]

    kmeans = KMeans(n_clusters=2, random_state=0).fit(mid_points)

    points_zero = []
    boxes_zero = []
    for i, point in enumerate(boxes):
        if kmeans.labels_[i] == 0:
            points_zero.append(names[i])
            boxes_zero.append(boxes[i])

    points_one = []
    boxes_one = []
    for i, point in enumerate(boxes):
        if kmeans.labels_[i] == 1:
           points_one.append(names[i])
           boxes_one.append(boxes[i])

    zeros = [(box, name) for box, name in zip(boxes_zero, points_zero)]
    ones = [(box, name) for box, name in zip(boxes_one, points_one)]

    zeros.sort(key=lambda x: x[0][0])
    ones.sort(key=lambda x:x[0][0])

    final = []
    zero = [z[1] for z in zeros]
    one = [z[1] for z in ones]

    cen_zeros_x = [(z[0][0] + z[0][2])/2 for z in zeros]
    cen_zeros_y = [(z[0][1] + z[0][3])/2 for z in zeros]
    zeros_heights = [z[0][3] - z[0][1] for z in zeros]
    zeros_heights = sum(zeros_heights)/len(zeros_heights)


    cen_ones_x = [(z[0][0] + z[0][2])/2 for z in ones]
    cen_ones_y = [(z[0][1] + z[0][3])/2 for z in ones]
    ones_heights = [z[0][3] - z[0][1] for z in ones]
    ones_heights = sum(ones_heights)/len(ones_heights)    


    cen_zeros_x = sum(cen_zeros_x)/len(cen_zeros_x)
    cen_zeros_y = sum(cen_zeros_y)/len(cen_zeros_y)

    cen_ones_x = sum(cen_ones_x)/len(cen_ones_x)
    cen_ones_y = sum(cen_ones_y)/len(cen_ones_y)

#    print(cen_zeros_x, cen_zeros_y, cen_ones_x, cen_ones_y)

    if cen_zeros_y > cen_ones_y :
        if cen_zeros_y > cen_ones_y + zeros_heights*0.5:
            ordered = one + zero
        else:
            if cen_zeros_x > cen_ones_x:
                ordered = one + zero #print(one+zero)
            else:
                ordered = zero + one #print(zero+one)
    else:
        if cen_ones_y > cen_zeros_y + ones_heights*0.5:
            ordered = zero + one
        else:
            if cen_ones_x > cen_zeros_x:
                ordered = zero + one
            else:
                ordered = one + zero

    return ''.join(ordered)


def drawImage(image, xml):
    boxxs = []
    names = []
    I = cv2.imread(image)
    objs = read_xml(xml)
    for obj in objs:
        boxes = objs[obj]

        for box in boxes:
            boxxs.append( [int(i) for i in box]) #(height, width, height/shape[0], width/shape[1])
            names.append(obj)
    ordered_string = getSortedBoxes(boxxs, names)
    return ordered_string

for image in images:
    print(image)
    xml_name = os.path.join(sys.argv[1], os.path.basename(image).replace('.jpg', '.xml'))
    image_name = image
    ordered_string = drawImage(image, xml_name)
    print(ordered_string)

