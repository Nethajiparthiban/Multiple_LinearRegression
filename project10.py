import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
#Read the data
data=pd.read_csv(r"D:\Git\Git-Projects\CarPrice_Assignment.csv")
Y=data.iloc[:,-1]
col_to_encode=['CarName','fueltype','aspiration','doornumber','carbody','drivewheel',
               'enginelocation','enginetype','cylindernumber','fuelsystem']
other_col=['car_ID','symboling','wheelbase','carlength','carwidth','carheight','curbweight',
           'enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm',
           'peakrpm','highwaympg']
X=data[col_to_encode+other_col]
categori_col=['CarName']
ordinal_cco=['fueltype','aspiration','doornumber','carbody','drivewheel',
             'enginelocation','enginetype','cylindernumber','fuelsystem']
#Determining the transformeres
ct=ColumnTransformer(transformers=[
    ('Car_Name_encoder',OneHotEncoder(),col_to_encode),
    ('others_encoder',OrdinalEncoder(),ordinal_cco)
],remainder='passthrough')
x=ct.fit_transform(X)
#training the data set
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)
#fitting to the data set
stud=LinearRegression()
stud.fit(x_train,y_train)
#predicting
y_pred=stud.predict(x_test)
#accuracy.
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),'%')