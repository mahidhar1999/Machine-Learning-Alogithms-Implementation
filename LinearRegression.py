import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
x=np.array([5,7,4,15,12,9])
y=np.array([8,9,12,26,16,13])
x=x.reshape(-1,1)
reg=linear_model.LinearRegression()
reg.fit(x,y)
print(reg.coef_)
print(reg.intercept_)
z=np.array([55])
z=z.reshape(-1,1)
print(reg.predict(z))
plt.xlabel('X', fontsize=20)
plt.ylabel('Y', fontsize=20)
plt.scatter(x,y)
plt.plot(x,reg.predict(x))