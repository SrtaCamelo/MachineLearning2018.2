#Aprendizado de máquina 2018.2
#Raissa Camelo
#ML_04_Pre-processamento

#Imports
import cv2
import os
import pandas as pd

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
    df = pd.DataFrame(data)
    df.columns = H0
    df.to_csv('h0_geometric.csv')

    print(df)
"""
param: img: Cv2 image
param: label: string
This function extracts features from the numpy based image.
"""
def extractFeatures(img, label):
    img_array = []
    #---Countours--------
    cnt, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours[0][0])
    #-------Area ---------
    area = cv2.contourArea(contours[0])
    #print(area)
    #-------Height and Weight------
    height, width = img.shape
    #print (height, width)
    #--------Perimeter----
    perimeter = cv2.arcLength(contours[0], True)
    #print(perimeter)
    #--------Number of Vertices----
    approx = cv2.approxPolyDP(contours[0], 0.04 * perimeter, True)
    approx = len(approx)
    print(approx)
    #--------Angle of each vertice-----
    #----------Test--------
    #cv2.imshow('IBAGE', cont)
    #cv2.waitKey()
    lab = labelGeo(label)
    h =[height,width,perimeter,area,approx,lab]
    h0 = [height,width,lab]
    h1 = [area,approx,perimeter,lab]


    I.append(h0)
    #h1 = [area,approx,lab]
    #print(h0)
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

    cv2.imshow("ibage", tre)
    cv2.waitKey()

    extractFeatures(tre, label)

    #cv2.imshow('TREXOLDED',tre)
    #cv2.waitKey()

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
#path = "2d_geometric_shapes_dataset"
#openfiles(path)

doTheMagic('./2d_geometric_shapes_dataset/squares/square_0061.jpg',"label")

#savePandas(I)

