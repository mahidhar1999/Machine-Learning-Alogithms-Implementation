import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
cancer=load_breast_cancer()
df=pd.DataFrame(cancer['data'])
scaler=StandardScaler()
scaler.fit(df)
scaled_data=scaler.transform(df)
pca=PCA(n_components=2)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
plt.xlabel('First principle component')
plt.ylabel('Second principle component')
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
