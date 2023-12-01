import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv(r"D:\Git\Git-Projects\User_Data.csv")
Y=data.iloc[:,-1]
ord_col=['Gender']
othe_col=['User ID','Age','EstimatedSalary']
X=data[ord_col+othe_col]
#print(X)
ct=ColumnTransformer(transformers=[('ord_encoder',OneHotEncoder(),ord_col)],remainder='passthrough')
x=ct.fit_transform(X)
#Training
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)
#fitting to algoritham.
stud=LinearRegression()
stud.fit(x_train,y_train)
#prediction
y_pred=stud.predict(x_test)
#accuracy
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),'%')