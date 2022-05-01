import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

import os
os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz/bin/'

new_world = pd.read_csv("data/data.csv")

print(new_world)

n = {'RU': 0, 'EN': 1}
new_world['Language'] = new_world['Language'].map(n)

t = {'Heal': 0, 'Tank': 1, 'DD': 2}
new_world['Specialization'] = new_world['Specialization'].map(t)

print(new_world)

feature_cols = ['Specialization', 'Ilevel', 'Rating', 'Language']
X = new_world[feature_cols]
y = new_world.Y 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

clf = DecisionTreeClassifier()

clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("Точность:",metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('new_world.png')
Image(graph.create_png())

row = pd.DataFrame([[2, 299, 2000, 0]],columns=['Specialization', 'Ilevel', 'Rating', 'Language'],dtype=float)
print(clf.predict(row))