#Machine Learning L02
#DataSet Pre-processing
#SrtaCamelo

#----------------Imports--------------------
import pandas as pd
import csv
from sklearn import preprocessing

#-------------Function Definitions----------
#----------------Open File Function------------------
def openFile(path):
    f = open(path, 'r+')
    my_file_data = f.read()
    f.close()
    return my_file_data

#---------------List to Tuple-------------------
def tuplenizer(list):
    outlist = []
    for i in range(len(list)):
        tuple = (i,list[i])
        outlist.append(tuple)
    return outlist
#--------------Change Labels to numbers---------
def numberize(column,tuple):
    new_column = []
    for item in column[1:]:
        for name in tuple:
            if item == name[1]:
                new_column.append(name[0])
    return new_column
#---------------Remove White Spaces-----------------
def remove_white_sp(infile,outfile):
    with open(infile) as infile, open(outfile, 'a') as outfile:
        for line in infile:
            line = ' '.join(line.split())
            line+='\n'
            outfile.write(line)

#-------------txt to csv function-------------------
def txt_2_csv(infile,outfile, header):
    with open(outfile,'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(header)
    with open(infile) as infile, open(outfile, 'a') as outfile:
        for line in infile:
            outfile.write(line.replace(' ', ','))

#------------------Remove A row function--------------

#-----------------Discretize Atributes (hot-one)----------------
def discretizer(data,column_to_disc):
    dummies = pd.get_dummies(data[column_to_disc])
    data = data.merge(dummies, left_index=True, right_index=True)
    del data[column_to_disc]
    return data

#----------------Discretie Atributes (numbered)----------------
def discretizer2(data, column_to_disc,class_tuple):
    new_column = numberize(data[column_to_disc],class_tuple)
    new_column = pd.DataFrame({'clas':new_column})
    data = data.merge(new_column, left_index=True, right_index=True)
    del data[column_to_disc]
    return data

#-------------------Normalize 0 -1 range---------------
def normalizer(data,atributes, final_column):
    #normalized_df = (data - data.min()) / (data.max() - data.min())
    for atribute in atributes:
        x = data[[atribute]].values.astype(float)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_normalized = pd.DataFrame(x_scaled)
        #print(df_normalized)
        data = data.merge(df_normalized,left_index=True, right_index=True)
        del data[atribute]
    data.columns = final_column
    return data
    #print(dt_amp)
#------------------Save Data Back into csv------------

def saveCSV(data,filename):
    data.to_csv(filename, header= True, mode='w',sep=',')
#--------------Main Code--------------------


#----------------IRIS -----------------------

    #-------Headers $ Columns List--------------
path = "C:/Users/SrtaCamelo/Documents/2018.2/Machine_Pacifico/ML-L02/DataSets/original/iris_dataSet.txt"
header = ['sepal length', 'sepal width','petal length', 'petal width','class']
atributes_to_normalize = ['sepal length', 'sepal width','petal length', 'petal width']
final_column = ['class','sepal length','sepal width','petal length','petal width']
classes = ['Iris-setosa','Iris-versicolor','Iris-virginica']
class_tuple = tuplenizer(classes)

    #--------OutPut File names------------------
filename = "iris_dataSet.csv"
pcd_filename = 'C:/Users/SrtaCamelo/Documents/2018.2/Machine_Pacifico/ML-L02/DataSets/processed/numberized/iris_pcd_dataSet.csv'

#--file = openFile(path)
txt_2_csv(path,filename,header)

data = pd.read_csv(filename)
data = normalizer(data,atributes_to_normalize,final_column)
data = discretizer2(data,'class',class_tuple)

saveCSV(data,pcd_filename)


#------------------Shuttle Statlog-------------------
    #-------Headers $ Columns List--------------
path = "C:/Users/SrtaCamelo/Documents/2018.2/Machine_Pacifico/ML-L02/DataSets/original/statlog_dataSet.txt"
header = ['time','atribute1','atribute2','atribute3','atribute4','atribute5','atribute6','atribute7','atribute8','class']
atributes_to_normalize = ['time','atribute1','atribute2','atribute3','atribute4','atribute5','atribute6','atribute7','atribute8']
final_header = ['class','time','atribute1','atribute2','atribute3','atribute4','atribute5','atribute6','atribute7','atribute8']

    #--------OutPut File names------------------
filename = "statlog_dataSet.csv"
pcd_filename = 'C:/Users/SrtaCamelo/Documents/2018.2/Machine_Pacifico/ML-L02/DataSets/processed/numberized/statlog_pdc_dataSet.csv'

txt_2_csv(path,filename,header)
data = pd.read_csv(filename)
data = normalizer(data,atributes_to_normalize,final_header)

saveCSV(data,pcd_filename)


#------------------Yast-------------------
    #-------Headers $ Columns List--------------
path = "C:/Users/SrtaCamelo/Documents/2018.2/Machine_Pacifico/ML-L02/DataSets/original/yeast_dataSet.txt"
header = ['seq_name','mcg','gvh','alm','mit','erl','pox','vac','nuc','class']
atributes_to_normalize = ['mcg','gvh','alm','mit','erl','pox','vac','nuc']
final_header = ['class','mcg','gvh','alm','mit','erl','pox','vac','nuc']

classes = ['CYT','NUC','MIT','ME3','ME2','ME1','EXC','VAC','POX','ERL']
class_tuple = tuplenizer(classes)
    #--------OutPut File names------------------
txtfile = "yeast_dataSet_wsr.txt"
filename = "yeast_dataSet.csv"
pcd_filename = 'C:/Users/SrtaCamelo/Documents/2018.2/Machine_Pacifico/ML-L02/DataSets/processed/numberized/yeast_pdc_dataSet.csv'

 #Remove White Spaces First
remove_white_sp(path,txtfile)
txt_2_csv(txtfile,filename,header)
data = pd.read_csv(filename)

del data['seq_name']

data = normalizer(data,atributes_to_normalize,final_header)

data = discretizer2(data,'class',class_tuple)
saveCSV(data,pcd_filename)
