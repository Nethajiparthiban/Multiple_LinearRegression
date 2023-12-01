import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv("D:\Git\Git-Projects\Student_Performance.csv")
cat_col=['Extracurricular Activities']
other_col=['Hours Studied','Previous Scores','Sleep Hours','Sample Question Papers Practiced']
X=data[cat_col+other_col]
Y=data.iloc[:,-1]
#print(X.columns)
ct=ColumnTransformer(transformers=[('cat_encoder',OneHotEncoder(),cat_col)],remainder='passthrough')
x=ct.fit_transform(X)
#print(x)
#Train
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)
#fiting
stud=LinearRegression()
stud.fit(x_train,y_train)
#prediction
y_pred=stud.predict(x_test)
#accuracy
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),'%')