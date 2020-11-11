import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
df=pd.read_csv("C:/Users/mahid/ML LAB/Linear Regression/original_data.csv")
a=df['salinity']
b=df['temp']
data_train,data_test,tar_train,tar_test=train_test_split(a,b,test_size=0.5,random_state=10)
data_train1=data_train.values.reshape(-1,1)
data_test1=data_test.values.reshape(-1,1)
tar_train1=tar_train.values.reshape(-1,1)
tar_test1=tar_test.values.reshape(-1,1)
reg=LinearRegression()
reg.fit(data_train1,tar_train1)
polyreg=PolynomialFeatures(degree=3)
x_poly=polyreg.fit_transform(data_train1)
reg1=LinearRegression()
reg1.fit(x_poly,tar_train1)
plt.title("Salinity vs Temp")
plt.xlabel("salinity")
plt.ylabel("temp")
plt.scatter(df.salinity,df.temp,color="red",marker="+")
plt.plot(data_test1,reg1.predict(polyreg.fit_transform(data_test1)))
print("Error values for multiple regression")
print(reg1.predict(polyreg.fit_transform(data_test1))-tar_test1)
print("Error values for linear regression")
print(reg.predict(data_test1)-tar_test1)
plt.show()
