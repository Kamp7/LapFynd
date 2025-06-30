import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('laptop_dataset_final.csv')
print(df)

df.dropna(subset=['Price (Rs)'], inplace=True)

cols=['Brand','Operating System','Display Type','Display Features','Display Touchscreen','Processor','Graphic Processor','Capacity','SSD Capacity','Fingerprint scanner','Expandable Memory','Backlit Keyboard','HDD Capacity','Battery Life','Fast Charging Support']
x=df[cols]
y=df.iloc[:,-1]   

features1 = [
    'Operating System', 'Display Touchscreen', 'SSD Capacity','Fingerprint scanner', 'Fast Charging Support','Battery Life'
]
for feature in features1:
    x.fillna({feature:x[feature].mode()[0]},inplace=True)

x['Expandable Memory'].fillna("0 GB", inplace=True)
x['HDD Capacity'].fillna("0 TB", inplace=True)
x['Capacity'].fillna("0 TB", inplace=True)

features2= [
    'Display Type', 'Display Features', 'Graphic Processor', 'Backlit Keyboard', 'HDD Capacity'
]

for feature in features2:
    x.fillna({feature:"None"},inplace=True)

x= x.convert_dtypes()

metric=['Capacity','SSD Capacity','Expandable Memory','HDD Capacity','Battery Life']
nominal=['Brand','Operating System','Display Type','Display Features','Display Touchscreen','Processor','Graphic Processor','Fingerprint scanner','Backlit Keyboard','Fast Charging Support']

for var in metric:
    x[var] = (x[var].str.extract(r'([\d.]+)').astype(float))


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False),  nominal)
    ],
    remainder='passthrough'   
)
enc_x=ct.fit_transform(x)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(enc_x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test) 
np.set_printoptions(precision=2)
compare = pd.DataFrame({
    'Predicted': y_pred,
    'Actual':    y_test.to_numpy()  
})

print(compare)

from sklearn import metrics
meanAbErr = metrics.mean_absolute_error(y_test, y_pred)
meanSqErr = metrics.mean_squared_error(y_test, y_pred)
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('R squared: {:.2f}'.format(regressor.score(enc_x,y)*100))
print('Mean Absolute Error:', meanAbErr)
print('Mean Square Error:', meanSqErr)
print('Root Mean Square Error:', rootMeanSqErr)


