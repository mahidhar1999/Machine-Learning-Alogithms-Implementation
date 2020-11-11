import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("C:/Users/mahid/ML LAB/Logistic Regression/insurance_data.csv")
plt.scatter(df.age,df.bought_insurance)
data_train,data_test,tar_train,tar_test=train_test_split(df.age,df.bought_insurance,test_size=0.25,random_state=10)
data_train1=data_train.values.reshape(-1,1)
data_test1=data_test.values.reshape(-1,1)
tar_train1=tar_train.values.reshape(-1,1)
tar_test1=tar_test.values.reshape(-1,1)
model=LogisticRegression()
model.fit(data_train1,tar_train1)
print(model.predict(data_test1))
plt.xlabel("age")
plt.ylabel("bought or not")
plt.plot(data_test1,model.predict(data_test1))