import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
df=pd.read_csv("C:/Users/mahid/ML LAB/Naive bayes/titanic.csv")
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
target=df.Survived
inputs=df.drop(['Survived'],axis='columns')
intofsex=pd.get_dummies(inputs.Sex)
inputs=pd.concat([inputs,intofsex],axis='columns')
inputs.drop(['Sex'],axis='columns',inplace=True)
inputs.Age=inputs.Age.fillna(inputs.Age.mean())
data_train,data_test,tar_train,tar_test=train_test_split(inputs,target,test_size=0.25,random_state=10)
model=GaussianNB()
model.fit(data_train,tar_train)
print(model.score(data_test,tar_test))
print(model.predict(data_test)-tar_test)
