#Importing the modules
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
#reading the data set
data=pd.read_csv("D:\Git\Git-Projects\Ecommerce_Customers.csv")
Y=data.iloc[:,-1]
ordi_col=['Email','Address','Avatar']
other_col=['Avg. Session Length','Time on App','Time on Website','Length of Membership']
ct=ColumnTransformer(transformers=[(
    'ord_encoder',OrdinalEncoder(),ordi_col
)],remainder='passthrough')
X=data[ordi_col+other_col]
x=ct.fit_transform(X)
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=2)
#fitting algoritham
stud=LinearRegression()
stud.fit(x_train,y_train)
#predict the data
y_pred=stud.predict(x_test)
#accuracy..
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),"%")