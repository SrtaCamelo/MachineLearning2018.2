
#Machine Learning L02
#k-fold evaluation
#SrtaCamelo

#----------------Imports--------------------
from time import time
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


#--------------Structures for Score Validation----------
tree_acuracy = []
nb_acuracy = []
knn_1_acuracy = []
wknn_1_acuracy = []
knn_3_acuracy = []
wknn_3_acuracy = []
knn_5_acuracy = []
wknn_5_acuracy = []
svm_linear_acuracy = []
svm_rbf_acuracy = []
logi_acuracy = []
neural_acuracy = []

tree_time = []
nb_time = []
knn_1_time = []
wknn_1_time = []
knn_3_time = []
wknn_3_time = []
knn_5_time = []
wknn_5_time = []
svm_linear_time = []
svm_rbf_time = []
logi_time = []
neural_time = []

#------------- Define Functions----------
#---------------Gosset's hipotesis calculatiom---
def tstudant_hipotesis(a,b):
    t2, p2 = stats.ttest_ind(a, b)
    return t2,p2
#--------------Log Generator (Could have been saved in a txt file, but naaaah)--------------
def showHipotesis():
    print("T-STUDANT HIPOTESIS")
    print("KNN-1 e KNN-3")
    t2,p2 = tstudant_hipotesis(knn_1_acuracy,knn_3_acuracy)
    print("T-value:",t2,"P-value:",p2)
    print("KNN-1 e KNN-5")
    t2, p2 = tstudant_hipotesis(knn_1_acuracy, knn_5_acuracy)
    print("T-value:", t2, "P-value:", p2)
    print("KNN-3 e KNN-5")
    t2, p2 = tstudant_hipotesis(knn_3_acuracy, knn_5_acuracy)
    print("T-value:", t2, "P-value:", p2)

    print("WKNN-1 e WKNN-3")
    t2, p2 = tstudant_hipotesis(wknn_1_acuracy, wknn_3_acuracy)
    print("T-value:", t2, "P-value:", p2)
    print("WKNN-1 e WKNN-5")
    t2, p2 = tstudant_hipotesis(wknn_1_acuracy, wknn_5_acuracy)
    print("T-value:", t2, "P-value:", p2)
    print("WKNN-3 e WKNN-5")
    t2, p2 = tstudant_hipotesis(wknn_3_acuracy, wknn_5_acuracy)
    print("T-value:", t2, "P-value:", p2)

    print("SVM-LINEAR e SVM-RBF")
    t2,p2 = tstudant_hipotesis(svm_linear_acuracy,svm_rbf_acuracy)
    print("T-value:", t2, "P-value:", p2)
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

def prepareStatistics(List_ofLists):
    for acuracy_list in List_ofLists:
        #print(acuracy_list[0])
        showStatistics(acuracy_list)

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
def call_classifiers(x_train, y_train, x_test, y_test):
    start_time = time()

    #---------------Training Models------------
    #----------------Decision Tree----------
    tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
    max_depth=3, min_samples_leaf=5)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    tree_accu = accuracy_score(y_test, y_pred)
    tree_acuracy.append(tree_accu)

    t1 = round(time()- start_time, 3)
    tree_time.append(t1)
    #---------------Naive Bayes-------------
    start_time = time()

    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    gnb_predic = gnb.predict(x_test)
    gnb_accu = accuracy_score(y_test,gnb_predic)
    nb_acuracy.append(gnb_accu)

    t1 = round(time() - start_time, 3)
    nb_time.append(t1)
    #--------------KNN - All variations-----
    start_time = time()

    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn1.fit(x_train,y_train)
    knn1_predic = knn1.predict(x_test)
    knn1_accu = accuracy_score(y_test,knn1_predic)
    knn_1_acuracy.append(knn1_accu)

    t1 = round(time() - start_time, 3)
    knn_1_time.append(t1)

    start_time = time()
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(x_train,y_train)
    knn3_predic = knn3.predict(x_test)
    knn3_accu = accuracy_score(y_test,knn3_predic)
    knn_3_acuracy.append(knn3_accu)

    t1 = round(time() - start_time, 3)
    knn_3_time.append(t1)

    start_time = time()
    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5.fit(x_train,y_train)
    knn5_predic = knn5.predict(x_test)
    knn5_accu = accuracy_score(y_test,knn5_predic)
    knn_5_acuracy.append(knn5_accu)
    t1 = round(time() - start_time,3)
    knn_5_time.append(t1)

    #--------------WKNN - All variations-----
    start_time = time()

    wknn1 = KNeighborsClassifier(n_neighbors=1,weights='distance')
    wknn1.fit(x_train, y_train)
    wknn1_predic = wknn1.predict(x_test)
    wknn1_accu = accuracy_score(y_test, wknn1_predic)
    wknn_1_acuracy.append(wknn1_accu)

    t1 = round(time() - start_time, 3)
    wknn_1_time.append(t1)

    start_time = time()

    wknn3 = KNeighborsClassifier(n_neighbors=3,weights='distance')
    wknn3.fit(x_train, y_train)
    wknn3_predic = wknn3.predict(x_test)
    wknn3_accu = accuracy_score(y_test, wknn3_predic)
    wknn_3_acuracy.append(wknn3_accu)

    t1 = round(time() - start_time, 3)
    wknn_3_time.append(t1)

    start_time = time()

    wknn5 = KNeighborsClassifier(n_neighbors=5,weights='distance')
    wknn5.fit(x_train, y_train)
    wknn5_predic = wknn5.predict(x_test)
    wknn5_accu = accuracy_score(y_test, wknn5_predic)
    wknn_5_acuracy.append(wknn5_accu)

    t1 = round(time() - start_time, 3)
    wknn_5_time.append(t1)

    #----------------SVM Linear--------------

    print("STARTED")
    start_time = time()

    svm_linear = svm.SVC(C=1, kernel='linear')
    svm_linear.fit(x_train, y_train)
    print("fited")
    svm_predic = svm_linear.predict(x_test)
    svml_accu = accuracy_score(y_test,svm_predic)
    svm_linear_acuracy.append(svml_accu)
    print("o")
    t1 = round(time() - start_time, 3)
    svm_linear_time.append(t1)

    # ----------------SVM RBF--------------
    
    start_time = time()

    svm_rbf = svm.SVC(C=1, kernel='rbf')
    svm_rbf.fit(x_train, y_train)
    svm_predic = svm_rbf.predict(x_test)

    svmr_accu = accuracy_score(y_test, svm_predic)
    svm_rbf_acuracy.append(svmr_accu)

    t1 = round(time() - start_time, 3)
    svm_rbf_time.append(t1)
    print("SVM DONE\n")

    # --------Logistic Regression-----------

    start_time = time()

    logi = LogisticRegression(random_state=0, solver='lbfgs',
                              multi_class='multinomial')
    logi.fit(x_train, y_train)
    logi_predic = logi.predict(x_test)

    logi_accu = accuracy_score(y_test, logi_predic)
    logi_acuracy.append(logi_accu)

    t1 = round(time() - start_time, 3)
    logi_time.append(t1)
    #print("Logi done\n")

    #-------MLP----------------------
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    start_time = time()

    model = Sequential()
    #Architeture
    model.add(Dense(12, input_dim=5, activation='softmax'))
    model.add(Dense(9, activation='softmax'))
    #Compilation
    #'categorical_crossentropy'
    #‘softmax’
    model.compile(optimizer='sgd', loss='categorical_crossentropy',metrics=['accuracy'])
    #Fiting
    history = model.fit(x_train, y_train, batch_size=10, epochs=100)
    #Evaluation
    loss, accuracy = model.evaluate(x_test, y_test)
    neural_acuracy.append(accuracy)

    t1 = round(time() - start_time, 3)
    neural_time.append(t1)

    #print(accuracy)


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
#--------------K-fold method (k = 10)---------------------
def kfold_call(data,n_start, n_final,key):

    #---------Split Class column from features-------------
    X = data.iloc[:,n_start:n_final]
    Y = data[key]


    kf = KFold(n_splits=10, shuffle=False)
    for train_index, test_index in kf.split(X):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        call_classifiers(x_train, y_train, x_test, y_test)

#---------------Variables and Constants ----------------
path = "h_geometric_n.csv"

#---------------------Main Code ------------------------

for i in range(5):
    data = prepareDataSet(path)

    kfold_call(data,1,6,"Class")

#--------------These code lines above Do de SD, median and mean calculus---------------------------

List_ofLists =[(tree_acuracy,"Tree"),(nb_acuracy,"Naive bayes"),(knn_1_acuracy,"KNN -1"),(knn_3_acuracy,"KNN -3"),
               (knn_5_acuracy, "KNN -5"),(wknn_1_acuracy,"WKNN-1"),(wknn_3_acuracy,"WKNN-3"),(wknn_5_acuracy,"WKNN-5"),
               (svm_linear_acuracy,"SVM-Linear"),(svm_rbf_acuracy,"SVM-RBF"),(logi_acuracy,"Logistic Regression"),(neural_acuracy,"MLP")]


List_ofListx =[(tree_time,"Tree"),(nb_time,"Naive bayes"),(knn_1_time,"KNN -1"),(knn_3_time,"KNN -3"),
               (knn_5_time, "KNN -5"),(wknn_1_time,"WKNN-1"),(wknn_3_time,"WKNN-3"),(wknn_5_time,"WKNN-5"),
               (svm_linear_time,"SVM-Linear"),(svm_rbf_time,"SVM-RBF"),(logi_time,"Logistic Regression"),(neural_time,"MLP")]
print("ACURACY\n")
prepareStatistics(List_ofLists)
print("Execution TIME\n")
prepareStatistics(List_ofListx)

#-------------This call above calls t-studant hipotesis for classifiers variations-----------------
showHipotesis()
