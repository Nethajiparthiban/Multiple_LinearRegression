import pandas as pd
import numpy as np
df=pd.DataFrame({
    'experience':[np.NaN,np.NaN,'five','two','seven','three','ten','eleven'],
    'test_score(out of 10)':[8,8,6,10,9,7,np.NaN,7],
    'interview_score(out of 10)':[9,6,7,10,6,10,7,8],
    'salary($)':[50000,45000,60000,65000,70000,62000,72000,80000]
})
#filling Nan values
df['experience']=df.experience.fillna('zero')
df['test_score(out of 10)']=df['test_score(out of 10)'].fillna(np.ceil(df['test_score(out of 10)'].mean()))
from word2number import w2n
df['experience']=df['experience'].apply(w2n.word_to_num)
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn import linear_model
reg=linear_model.LinearRegression()
reg.fit(x,y)
reg.coef_#[2812.95487627 1845.70596798 2205.24017467]
reg.intercept_#17737.26346433768
reg.predict([[0,8,9]])#[52350.0727802]
#Cross validiation
#y=mx+b ie m1*x1+m2*x2+m3*x3+b
#2812.95487627*0+1845.70596798*8+2205.24017467*9+17737.26346433768##52350.07278020769
#lets predict the requested values
df1=pd.DataFrame({
    'experience':[2,12],
    'test_score(out of 10)':[9,10],
    'interview_score(out of 10)':[6,10]
})
w=reg.predict(df1)
df1['salary($']=w

#pickling
import pickle

with open('multi1_pickle','wb') as f:
    pickle.dump(reg,f)
with open('multi1_pickle','rb') as k:
    multi1=pickle.load(k)
print(multi1.predict([[2,9,6]]))