import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.model_selection import train_test_split
df=pd.read_csv("C:/Users/mahid/ML LAB/salaries.csv")
inputs=df.drop('salary_more_then_100k',axis='columns')
target=df['salary_more_then_100k']
le_company=LabelEncoder()
le_job=LabelEncoder()
le_degree=LabelEncoder()
inputs['company_n']=le_company.fit_transform(inputs['company'])
inputs['job_n']=le_job.fit_transform(inputs['job'])
inputs['degree_n']=le_degree.fit_transform(inputs['degree'])
inputs_n=inputs.drop(['company','job','degree'],axis='columns')
data_train,data_test,tar_train,tar_test=train_test_split(inputs_n,target,test_size=0.2,random_state=10)
model=tree.DecisionTreeClassifier()
model.fit(data_train,tar_train)
print(model.predict(data_test))
print(tar_test)
print(model.score(data_test,tar_test))