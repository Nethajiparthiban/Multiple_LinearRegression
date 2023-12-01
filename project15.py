import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data=pd.read_csv("D:\Git\Git-Projects\Music.csv")
Y=data.iloc[:,-1]
X=data[['SongLength','NumInstruments','Tempo','LyricalContent','ReleasedYear']]
#print(Y)
#traing
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
#Fitting to the data set
stud=LinearRegression()
stud.fit(x_train,y_train)
#predicting
y_pred=stud.predict(x_test)
#accuracy
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),'%')
