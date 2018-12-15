#---Just a testing File---

#IMPORTS
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import pandas as pd
import cv2

#CODE
def prepareDataSet(datapath):
    #Shuffle
    data = pd.read_csv(datapath)
    data = data.sample(frac=1)
    X = data.iloc[:,1:6]
    Y = data['Class']
    Y = pd.get_dummies(Y)
    x_train, x_test = X.iloc[1:800], X.iloc[801:1071]
    y_train, y_test = Y.iloc[1:800], Y.iloc[801:1071]

    return x_train, x_test,y_train, y_test

def neuralNetwork(x_train, x_test,y_train, y_test):
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
    print(accuracy)

#main
path = "h_geometric.csv"
x_train, x_test,y_train, y_test = prepareDataSet(path)
neuralNetwork(x_train, x_test,y_train, y_test)