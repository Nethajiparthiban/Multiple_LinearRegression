import numpy as np
import pandas as pd
df=pd.DataFrame({
    'area':[2600,3000,3200,3600,4000],
    'bedrooms':[3,4,np.NaN,3,5],
    'age':[20,15,18,30,8],
    'price':[550000,565000,610000,595000,76000]
})
#Filling the Nan values

df['bedrooms']=df['bedrooms'].fillna(np.floor(df.bedrooms.median()))
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(x,y)
reg.coef_#[   -153.45 -106395.      8565.  ]
reg.intercept_#1209654.9999999988
reg.predict([[3300,3,8]])#[452605.]
#cross validation
#y=mx+b ie m1*x1+m2*x2+m3*x3+b
#print(3300*-153.45+3*-106395+8*8565+1209654.9999999988)
#pridiction should done for the below
df1=pd.DataFrame({
    'area':[3800,2500],
    'bedrooms':[4,3],
    'age':[40,5]
})
z=reg.predict(df1)
df1['price']=z
df2=pd.concat([df,df1],ignore_index=True)

import pickle

with open('multi2_pickle','wb') as f:
    pickle.dump(reg,f)

with open('multi2_pickle','rb') as k:
    multi2=pickle.load(k)

print(multi2.predict([[3800,4,40]]))