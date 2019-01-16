#Mineração de Textos 2018.2
#Clustering News
#Raissa Camelo

import pandas as pd
import numpy as np
import random

#---------Lists--------
kmeans_wss = []
ga_wss = []
hibrid_wss = []
kmeans_majory = []
ga_majory = []
hibrid_majory = []
#---------Define Functions---
#--------------Shuffle data an Split Features-------------
"""
Param: dapath: String with the file path
Return: pandas Dataframe
Open dataset and returns it as a pandas Frame
"""
def prepareDataSet(datapath):
    #Shuffle
    data = pd.read_csv(datapath)
    data = data.sample(frac=1)
    del data['class']
    return data
def wss_calculate(center,data):
    wss = 0
    return wss
def accuMajory_calculate():
    amajo = 0
"""
Param: Data: Pandas Data Frame -> DataSet
       n_clusters: Integer -> Number of classes in the dataSet
Initializes the population (size 20) of a given data set
"""
def fetch_population(data, n_clusters):
    size = data.shape[0]
    n_features = len(data.columns)
    #print(n_features)
    #data_indexes = np.arange(size)
    popList = []
    for i in range(40):
        n_ple = random.sample(range(0, size), n_clusters)
        popList.append(n_ple)
    population = []
    for center in popList:
        for i in range(n_clusters):
            index = center[i]
            print(index)
            element = 'e'

    #return population
def ga_call(n_generations,data):
    ga = "l"
    population = fetch_population(data,3)

def kmeans(data, center):
    a = 0

#-----------------MAIN-------------------
path = "teste100.csv"

data = prepareDataSet(path)
print(len(data.columns), "COLUNAS")
#print(data)
#----------- K-MEANS

#
#ga_call(100,data)
#fetch_population(data,3)