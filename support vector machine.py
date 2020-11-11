import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
plt.title("svm model")
plt.xlabel("Length")
plt.ylabel("Width")
df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]
#plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color="red")
#plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color="blue")
#plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'],color="green")
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color="red")
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color="blue")
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color="green")
x=df.drop(['target'],axis="columns")
y=df.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
model=SVC()
model.fit(x_train,y_train)
print(model.fit(x_train,y_train))
#can tune parameters and tune score
print(model.score(x_test,y_test))
a1=model.predict(x_test)-y_test
