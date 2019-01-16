#Aprendizado de máquina 2018.2
#Raissa Camelo
#ML_04_Pre-processamento

#Imports
import cv2
import os
import pandas as pd
import numpy as np
import imutils
from imutils import perspective
from scipy.spatial import distance as dist
from sklearn.preprocessing import StandardScaler


#Type Key
key = [("circles", 0),("ellipses",1),("hexagons",2),("lines",3),
       ("rectangles",4),("rhombuses",5),("squares",6),("trapezia",7),("triangles",8)]
#Headers
H = ['Altura','Largura','Perimetro','Area','N_vertices','Class']
H0 = ['Altura','Largura','Class']
H1 = ['Area','N_vertices','Perimetro','Class']
#Numberized Image Array
I = []
"""
param: label: string
returns: n: integer
This function receives a key (geometric figure) and returns its number (0-7)
to numberize the classes of the data.
"""
def labelGeo(label):
    for k in key:
        if label == k[0]:
            n = k[1]
            return n
def savePandas(data):
    #save_path = 'h_geometric.csv'
    df = pd.DataFrame(data)
    df.columns = H
    df = normalizer(df,H)
    df.to_csv('h_geometric_n.csv')

    print(df)

#-------------------Normalize 0 -1 range---------------
def normalizer(data, final_col):
    scaler = StandardScaler()
    y = data.loc[:, ['Class']]
    dataframe = pd.DataFrame(scaler.fit_transform(data), columns=final_col)
    dataframe['Class'] = y['Class']
    print(dataframe)
    return dataframe

"""
params: 2-sized arrays, (euclidian points (x,Y)
Function calculates midpoint between two given points
"""
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
"""
param: img: Cv2 image
param: label: string
This function extracts features from the numpy based image.
"""
def extractFeatures(img, label):

    #---Countours--------
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours[0][0])

    #-------Area ---------
    area = cv2.contourArea(contours[0])
    #print(area)

    #-------Height and Weight------
    #cnts = contours.sort()
    for c in contours:
        if cv2.contourArea(c) < 100:
            continue
        # compute the rotated bounding box of the contour

        box = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        box = np.array(box, dtype="int")

        box = perspective.order_points(box)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)

        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        # compute the Euclidean distance between the midpoints
        height = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        width = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        #print(height, width)

    #--------Perimeter----
    perimeter = cv2.arcLength(contours[0], True)
    #print(perimeter)

    #--------Number of Vertices----
    approx = cv2.approxPolyDP(contours[0], 0.04 * perimeter, True)
    approx = len(approx)
    #print(approx)

    #--------Angle of each vertice-----

    lab = labelGeo(label)
    h =[height,width,perimeter,area,approx,lab]
    h0 = [height,width,lab]
    h1 = [area,approx,perimeter,lab]

    #print(h)
    I.append(h)

    #print(h0)
    # ----------Test--------
    # cv2.imshow('IBAGE', cont)
    # cv2.waitKey()
"""
Param: img_path: Relative path of each image
return: void
Function does the preprocessing steps with a passed image
RGB to Gray, Gaussian Blur and Binary Threshold
"""
def doTheMagic(img_path,label):
    img = cv2.imread(img_path)

    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray_image, (3, 3), 0)
    ret, tre = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

    #edged = cv2.Canny(tre, 50, 100)

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(tre, kernel, iterations=2)
    erosion = cv2.erode(dilation, kernel, iterations=2)

    extractFeatures(erosion, label)


    """
    cv2.imshow("ibage", erosion)
    cv2.waitKey()
    
    cv2.imshow('Dela',dilation)
    cv2.waitKey()

    cv2.imshow('ero', erosion)
    cv2.waitKey()
    """
def openfiles(path):
    folders = os.listdir(path)
    for fold in folders:
        way = path+'/'+fold
        images = os.listdir(way)
        for img in images:
            jpg = way+'/'+img
            doTheMagic(jpg,fold)
    #print(files)

#---------Main-----------
path = "2d_geometric_shapes_dataset"
openfiles(path)
savePandas(I)
#-----Normalize---------
#path = "h_geometric.csv"
#data = pd.read_csv(path)

#final_col = ['Altura','Largura','Perimetro','Area','N_vertices','Class']

#data = normalizer(I,final_col)
#savePandas(data)




