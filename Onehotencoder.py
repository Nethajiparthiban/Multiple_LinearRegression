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
#print(ohe_transform)
new_df=pd.concat([ohe_transform,df],axis='columns')
#print(new_df.columns)
new=new_df.drop(['town_west windsor','town'],axis='columns')
#print(new)
x=new.iloc[:,:-1]
y=new.iloc[:,-1]
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(x,y)
#lets pridict value for monore township
#print(reg.predict([[1,0,2600]]))
#lets pridict value for west windsor township
#print(reg.predict([[0,0,2600]]))#[579723.71533004]
#cross validation
#y=mx+b
reg.coef_#[-40013.97548914 -14327.56396474    126.89744141]
reg.intercept_#249790.36766286375
#print(126.89744141*2600+249790.36766286375)579723.7153288638
#print(reg.score(x,y))
import pickle
with open('pick_encode','wb') as f:
    pickle.dump(reg,f)
with open('pick_encode','rb')as k:
    _pred=pickle.load(k)
print(_pred.predict([[0,0,2600]]))