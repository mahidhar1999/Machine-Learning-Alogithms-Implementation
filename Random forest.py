import pandas as pd
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
digits=load_digits()
df=pd.DataFrame(digits.data)
df['target']=digits.target
data_train,data_test,tar_train,tar_test=train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)
model=RandomForestClassifier(n_estimators=90)
model.fit(data_train,tar_train)
print(model.fit(data_train,tar_train))
print(model.score(data_test,tar_test))
ypredicted=model.predict(data_test)
mat=confusion_matrix(tar_test,ypredicted)
print(mat)