import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("C:/Users/mahid/ML LAB/Linear Regression/original_data.csv")
a=df['salinity']
b=df['temp']
data_train,data_test,tar_train,tar_test=train_test_split(a,b,test_size=0.2,random_state=10)
data_train1=data_train.values.reshape(-1,1)
data_test1=data_test.values.reshape(-1,1)
tar_train1=tar_train.values.reshape(-1,1)
tar_test1=tar_test.values.reshape(-1,1)
reg=LinearRegression()
reg.fit(data_train1,tar_train1)
print("slope=",reg.coef_,"intercept=",reg.intercept_)
plt.title("Salinity vs Temp")
plt.xlabel("salinity")
plt.ylabel("temp")
plt.scatter(df.salinity,df.temp,color="red",marker="+")
plt.plot(df.salinity,reg.predict(df[['salinity']]))
plt.show()
print("Error values")
print(reg.predict(data_test1)-tar_test1)