import pandas as pd
df=pd.read_csv(r"C:\Users\Netha\Ananconda_onlineclass\Mission learning\ML from codebasics\Machine Learning\Linear_regression\carprices.csv")
import matplotlib.pyplot as plt
plt.xlabel('Car_Age')
plt.ylabel('Mileage')
plt.title('Car Mileage based on age')
plt.scatter(df['Age(yrs)'],df['Mileage'],color='blue')
plt.show()
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression

reg=LinearRegression()
reg.fit(x_train,y_train)
#print(reg.predict(x_test))
#print(y_test)
#print(len(x_test))
#print(reg.score(x_train,y_train))
import pickle

with open('train_pickle','wb') as f:
    pickle.dump(reg,f)
with open('train_pickle','rb') as k:
    pic=pickle.load(k)

print(pic.predict(x_test))