import pandas as pd

df=pd.DataFrame({
    'town':['monore township','monore township','monore township','monore township','monore township',
           'west windsor','west windsor','west windsor','west windsor',
           'robinsville','robinsville','robinsville','robinsville',],
    'area':[2600,3000,3200,3600,4000,2600,2800,3300,3600,2600,2900,3100,3600],
    'price':[550000,565000,610000,680000,725000,585000,615000,650000,710000,575000,
            600000,620000,695000]
})
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(handle_unknown='ignore',sparse_output=False).set_output(transform='pandas')
ohe_transform=ohe.fit_transform(df[['town']])
new_df=pd.concat([ohe_transform,df],axis='columns')
#in order to avoid multi-coliner issue we drop town and one hot encoder columns
new_df=new_df.drop(['town','town_monore township'],axis='columns')
x=new_df.iloc[:,:-1]
y=new_df.iloc[:,-1]
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(x,y)
reg.coef_#[25686.4115244  40013.97548914   126.89744141]
reg.intercept_#209776.39217373414
reg.predict([[0,0,2600]])#[539709.7398409]
#cross validation
#y=mx+b
print(reg.score(x,y))