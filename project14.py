import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
data = pd.read_csv("D:\Git\Git-Projects\Music.csv")
Y=data.iloc[:,-1]
cat_col=['Genre']
oth_col=['SongLength','NumInstruments','Tempo','LyricalContent','ReleasedYear']
X=data[cat_col+oth_col]
ct=ColumnTransformer(transformers=[('Cat_encoder',OneHotEncoder(),cat_col)],remainder='passthrough')
x=ct.fit_transform(X)
#print(x)
#Training
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)
#Fitting
stud=LinearRegression()
stud.fit(x_train,y_train)
#prediction
y_pred=stud.predict(x_test)
#accuracy
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),'%')