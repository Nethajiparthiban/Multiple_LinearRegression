import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Read the dataset
data = pd.read_csv(r"D:\Git\Git-Projects\advertising.csv")

# Extract the column(s) you want to transform
columns_to_encode = ['City', 'Country', 'Ad Topic Line']
other_columns = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']

X = data[columns_to_encode + other_columns]
Y = data.iloc[:, -1]

# Identify the columns to apply different transformations
categorical_columns = ['City', 'Country']
ordinal_columns = ['Ad Topic Line']

# Define the transformers
ct = ColumnTransformer(
    transformers=[
        ('city_country_encoder', OneHotEncoder(), categorical_columns),
        ('topicline_encoder', OrdinalEncoder(), ordinal_columns)
    ],
    remainder='passthrough'
)

# Apply the transformation
x= ct.fit_transform(X)
#Traiing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.25,random_state=1)
#fitting to dataset
from sklearn.linear_model import LinearRegression
stud=LinearRegression()
stud.fit(x_train,y_train)
#predicting
y_pred=stud.predict(x_test)
#Accuracy
from sklearn.metrics import mean_squared_error,r2_score
rscore=r2_score(y_test,y_pred)
rs=rscore*100
print(rs.round(),'%')
