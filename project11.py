#Importing modules
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
#Reading the data set
data=pd.read_csv(r"D:\Git\Git-Projects\data.csv")
#print(data.isnull().sum())
Y=data.iloc[:,-1]
col_to_encode=['diagnosis']
other_col=['id','radius_mean','texture_mean','perimeter_mean','area_mean',
           'smoothness_mean','compactness_mean','concavity_mean','concave points_mean',
           'symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se',
           'area_se','smoothness_se','compactness_se','concavity_se','concave points_se',
           'symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst',
           'area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst',
           'symmetry_worst','fractal_dimension_worst']
X=data[col_to_encode+other_col]
#encoding the dataset
ct=ColumnTransformer(transformers=[
    ('Diagnosis_encode',OneHotEncoder(),col_to_encode)
],remainder='passthrough')
x=ct.fit_transform(X)
#training the data set.
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)
#fitting to alogoritham
stud=LinearRegression()
stud.fit(x_train,y_train)
#predicting
y_pred=stud.predict(x_test)
#accuracy
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),'%')
