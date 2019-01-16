import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

file = "imdb_dataSet_pro.csv"
data = pd.read_csv(file)
print(data)

"""
tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
    max_depth=3, min_samples_leaf=5)
tree.fit(tfidf, y_train)
y_pred = tree.predict(x_test)
#tree_accu = accuracy_score(y_test, y_pred)
"""