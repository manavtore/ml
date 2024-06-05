import numpy as np
import pandas as pd
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

df = pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=['sepal length (cm)' ,'sepal width (cm)' , 'petal length (cm)' , 'petal width (cm)','target']);


X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'target']]
Y = df['target']

x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2,random_state=42)

kernelsarray = ['linear','poly','rbf']

for kernels in kernelsarray:
  model = SVC(kernel=kernels)
  model.fit(x_train,y_train)
  pred = model.predict(x_test)
  print(f"Accuracy using {kernels} kernel:{accuracy_score(y_test,pred)}")