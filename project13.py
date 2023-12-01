import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
data=pd.read_csv("D:\Git\Git-Projects\Ecommerce_Customers.csv")
#print(data)
Y=data.iloc[:,-1]
ordi_cl=['Email','Address','Avatar']
other_col=['Avg. Session Length','Time on App','Time on Website','Length of Membership']
X=data[ordi_cl+other_col]
ct=ColumnTransformer(transformers=[('ordi_encoder',OrdinalEncoder(),ordi_cl)],remainder='passthrough')
x=ct.fit_transform(X)
#print(x)
#training
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)
#fitting to algoritham
stud=LinearRegression()
stud.fit(x_train,y_train)
#predicting
y_pred=stud.predict(x_test)
#accuracy
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),'%')