#Mineração de Textos 2018.2
#Clustering News
#Raissa Camelo

#-----------------------Imports-------------------

from sklearn.model_selection import KFold
import pandas as pd
#Algorithms----------------------
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
#Validatin-----------------------
from sklearn import metrics
from sklearn.metrics import f1_score


#--------------Structures for Score Validation----------
dbscan_rand = []
spectral_rand = []
kmeans_rand = []

dbscan_fm = []
spectral_fm = []
kmeans_fm = []

dbscan_accu = []
spectral_accu = []
kmeans_accu = []
#---------------------Define Functions--------------

#----------Score Acuracy-----------------
"""
Parametros: lista contendo as 50 acuracias de teste
Retorno: Média, Desvio padrão e mediana das acurácias
"""
def scores(acuracy_list):
    #scaler = StandardScaler()
    series = pd.Series(acuracy_list)
    media = series.mean()
    dp = series.std()
    mediana = series.median()
    return media,dp,mediana

def rand_index(labels_true, labels_pred):
    rand = metrics.adjusted_rand_score(labels_true, labels_pred)
    return rand

#---------Função Fuleca só pra gerar o Log-----
def showStatistics(acuracy_list):
    mean, dp, median = scores(acuracy_list[0])
    print("###-------Algoritm:-----###",acuracy_list[1],"\n")
    print("#---Mean: ", mean,"\n")
    print("#---DP:", dp,"\n")
    print("#---Median:", median,"\n")

#--------------Função extramente (e desnecessariamente) redundante que chama todos os classificadores-------
"""
Parametros: x_train (conjunto de features de cada case para treino)
            y_train (classificação de cada case do treino)
            x_test (conjunto de features de cada case para teste)
            y_test (classificação de cada case do teste)
"""
#------------- Run all Algoritms for given dataSet------------
def call_classifiers(X,Y):
    #------------Spectral  Clustering ------------------------
    spec = SpectralClustering(affinity='rbf', assign_labels='discretize', coef0=1,
                       degree=3, eigen_solver=None, eigen_tol=0.0, gamma=1.0,
                       kernel_params=None, n_clusters=2, n_init=10, n_jobs=None,
                       n_neighbors=10, random_state=0)
    spec.fit(X,Y)
    matrix = spec.fit_predict(X, y=None)
    #print("SPECTRAL LABELS")
    #print(matrix , "\n\n")
    rand = rand_index(Y,matrix)
    spectral_rand.append(rand)

    fm = f1_score(Y, matrix, average='macro')
    spectral_fm.append(fm)

    """
    #-------------DBScan Clustering---------------------------
    dbs = DBSCAN(eps=0.5, min_samples=4)
    dbs.fit(x_train, y_train)
    matrix = dbs.fit_predict(x_train, y=None)
    rand = rand_index(y_train, matrix)
    #print("DBScan LABELS")
    #print(matrix , "\n\n")
    """

    #--------------------------K-Means-------------------------
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(X,Y)
    matrix = kmeans.fit_predict(X,Y)

    rand = rand_index(Y, matrix)
    kmeans_rand.append(rand)

    fm = f1_score(Y, matrix, average='macro')
    kmeans_fm.append(fm)




#--------------Shuffle data an Split Features-------------
def prepareDataSet(datapath):
    #Shuffle
    data = pd.read_csv(datapath)
    data = data.sample(frac=1)
    return data
"""
Parametros: data : Pandas DataFrame
Essa função divide o dataSet randomizado em 10 folds,
repassando o conjunto de treinamento e teste pros classificadores.

"""
#--------------Pre-Classification Method method (k = 10)---------------------
def cluster_call(data,n_start, n_final,key):

    #---------Split Class column from features-------------

    X = data.iloc[:,n_start:n_final]
    Y = data[key]
    call_classifiers(X,Y)

#---------------Variables and Constants ----------------
path = "subjects_data.csv"

#---------------------Main Code ------------------------
data = prepareDataSet(path)
cluster_call(data,2,len(data.columns)-1,"class")


#--------------These code lines above Do de SD, median and mean calculus---------------------------

"""
List_ofLists =[(kmeans_rand,"K-MEANS"),(spectral_rand,"Spectral")]

for acuracy_list in List_ofLists:
    print(acuracy_list[0])
    showStatistics(acuracy_list)
"""

print("k-MEANS, FM:")
print( kmeans_fm)
print("Spectral, FM:")
print(spectral_fm)

print("k-MEANS, RAND:")
print( kmeans_rand)
print("Spectral, RAND:")
print(spectral_rand)
