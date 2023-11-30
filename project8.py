import pandas as pd
data=pd.read_csv(r"D:\Git\Git-Projects\stat_females.csv")
Y=data.iloc[:,:1]
X=data.iloc[:,1:3]
#print(X)
#training
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
#fitting the data set
from sklearn.linear_model import LinearRegression
stud=LinearRegression()
stud.fit(x_train,y_train)
#predicting the data set
y_pred=stud.predict(x_test)
#accuracy
from sklearn.metrics import mean_squared_error,r2_score
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),'%')