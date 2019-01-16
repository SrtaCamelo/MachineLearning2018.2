import pandas as pd
import functions as fc
from sklearn.cluster import KMeans
import numpy as np
from random import randint
from pso_code import pso
import scipy.spatial.distance as sp

KCLUSTERS = 2


def WCSS(distance):
    retorno = []
    for x in distance:
        soma = 0
        for z in x:
            soma += z ** 2
        retorno.append(soma)

    return retorno
def WCSS2(distance):
    retorno = []
    soma = 0
    for z in distance:
        soma += z ** 2
        retorno.append(soma)
    return retorno



def break_vectors(vector):
    global KCLUSTERS
    retorno = np.split(vector, KCLUSTERS)
    return retorno


def wcssGenetic(centers):
    global KCLUSTERS
    array = break_vectors(centers)

    kmeans = KMeans(KCLUSTERS, init=pd.DataFrame(array), max_iter=1, n_init=1)
    kmeans.fit(pd.DataFrame(df_final))

    return kmeans.inertia_


def generatepopulation(X, numberpopu, K, rng):
    population = []

    for x in range(numberpopu):
        first = rng.permutation(X.shape[0])[:K]
        population.append(np.concatenate(X[first]))

    return population
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
"""
Mesure Genectic Algoritm alone
"""
def mesureGenetic(centers,data):
    labelList = []
    disList = []
    for individual in data:
        label = 0
        distance1 = sp.euclidean(centers[0],individual)
        distance2 = sp.euclidean(centers[1], individual)
        if distance1 < distance2:
            disList.append(distance1)
            label = 0
        elif distance2 < distance1:
            label = 1
            disList.append(distance2)
        else:
            label = 0
            disList.append(distance1)
        labelList.append(label)

    return centers, labelList, disList
def prepareDataSet(datapath):
    #Shuffle
    data = pd.read_csv(datapath)
    data = data.sample(frac=1)
    return data

data = prepareDataSet("teste100.csv")
lista = data['class']
#print(lista)
#del df_final['Class']

# df_final = pd.DataFrame(fc.PCAdataframe(50,df_final))

#df_final = fc.normalize(df_final)

# df_final['Class'] = df_finalaux['Class']


# print(len(df_finalaux['Class']))

del data['class']

df_final = np.array(data)
#print(df_final)

print("VAI COMEÇAR")
resultsKmeans = []  # ARRAY QUE SALVA AS ACURACIAS DO KMEANS
resultKmeansHybrid = []  # ARRAY QUE SALVA AS ACURACIAS DO HYBRID
acuraciaKmeans = []  # ARRAY QUE SALVA O WCSS DO K MEANS
acuraciaHybrid = []  # ARRAY QUE SALVA O WCSS DO HYBRID

resultGenetic = []
acuraciaGenetic = []

for y in range(10):

    print("RODADA " + str(y))
    resultparcial = []
    distance = []
    h = randint(1, 10)
    rng = np.random.RandomState(h)

    print("-----------------------------------------------")
    for algumacoisa in range(20):  # AQUI RODA O KMEANS 20 VEZES
        print("VEZ KMEANS " + str(algumacoisa))
        centers, labels, distance = fc.find_clusters(df_final, KCLUSTERS, rng,
                                                     100)  # AQUI VOCE PASSA 4 PARAMETROS, O SEU DATAFRAME EM UM NUMPYARRAY, O NUMERO DE CLASSES, O RNG QUE É A SEED E O NUMERO MAX DE ITERAÇÕES
        print(fc.accuracy_majority_vote(df_final, labels, lista,
                                        2))  # PRINTA O VOTO MAJORITARIO, QUE RECEBE SEU DATAFRAME EM FORMA DE ARRAY, OS LABELS DO RETORNO DE CIMA, E A LISTA QUE É A COLUNA DE LABELS DO SEU DATAFRAME ANTES DE VIRAR NUMPYARRAY
        acuraciaKmeans.append(fc.accuracy_majority_vote(df_final, labels, lista, 2))
        retorno = WCSS(distance)
        resultparcial.append(retorno[len(retorno) - 1])

    resultparcial = np.sort(resultparcial)

    resultsKmeans.append(resultparcial[0])
    
    population = generatepopulation(df_final, 20, KCLUSTERS, rng)

    p = pso(20, wcssGenetic, 0, 500, 2162, 100, init=population)
    array = np.array(p.get_Gbest())

    array = np.split(array, 2)
    print(array)

    print("Hybrid:")
    cen, lbl, dis = fc.find_clustersgenetic(df_final, KCLUSTERS, 100, array)
    print(fc.accuracy_majority_vote(df_final, lbl, lista, 2))
    acuraciaHybrid.append(fc.accuracy_majority_vote(df_final, lbl, lista, 2))

    ret = WCSS(dis)
    resultKmeansHybrid.append(ret[len(ret) - 1])

    # num = retorno[len(retorno) - 1]
    # dictionary[h] = num

    print("Genetic")
    population = generatepopulation(df_final, 20, KCLUSTERS, rng)
    p = pso(20, wcssGenetic, 0, 500, 2162, 100, init=population)
    array = np.array(p.get_Gbest())
    
    array = np.split(array, 2)
    #print(array)


    cen, lbl, dis = mesureGenetic(array,df_final)
    print(fc.accuracy_majority_vote(df_final, lbl, lista, 2))
    acuraciaGenetic.append(fc.accuracy_majority_vote(df_final, lbl, lista, 2))
    ret = WCSS2(dis)
    resultGenetic.append(ret[len(ret) - 1])

#
# print("Distancias" ,distance)
resultKmeansHybrid = np.sort(resultKmeansHybrid)
resultsKmeans = np.sort(resultsKmeans)


mediaw_h,dpw_h,medianaw_h = scores(resultKmeansHybrid)
mediaw_kmeans,dpw_kmeans,medianaw_kmeans = scores(resultsKmeans)
mediaw_g, dpw_g,medianaw_g = scores(resultGenetic)

media_h,dp_h,mediana_h = scores(acuraciaHybrid)
media_kmeans,dp_kmeans,mediana_kmeans = scores(acuraciaKmeans)
media_g, dp_g,mediana_g = scores(acuraciaGenetic)

print("---------WSS----------")
print("Media Hibrido", mediaw_h)
print("Media KMEANS", mediaw_kmeans)
print("Media Genetic", mediaw_g)

print("MEDIANA")
print(medianaw_kmeans, "MD KMEANS")
print(medianaw_h, "MD KHYBRID")
print(medianaw_g, "MD Genetic")

print("DP")
print(dpw_kmeans,"DP KMEANS")
print(dpw_h,"DP HYBRID")
print(dpw_g,"DP Genetic")

print("ACURACIA")

print("MEDIA")
print(media_kmeans, "ACURACIA KMEANS")
print(media_h, "ACURACIA KHYBRID")
print(media_g,"ACU GENETIC")

print("MEDIANA")
print(mediana_kmeans, "MD KMEANS")
print(mediana_h, "MD KHYBRID")
print(mediana_g,"MDGENETIC")

print("DP")
print(dp_kmeans,"DP KMEANS")
print(dp_h,"DP KMEANS")
print(dp_g,"DP GENETIC")

