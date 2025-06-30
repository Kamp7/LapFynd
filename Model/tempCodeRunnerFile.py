import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('laptop_dataset_final.csv')
#print(df['Thickness'].value_counts()/(len(df)-df['Thickness'].isnull().sum())*100)
#df['Thickness'].fillna(df['Thickness'].mode()[0], inplace=True)
#print(df['Thickness'].mode())
print(df.columns)
cols=['Brand','Operating System','Display Type','Display Features','Display Touchscreen','Processor','Graphic Processor','Capacity','SSD Capacity','Web-cam','Fingerprint scanner','Face Recognition','Expandable Memory','Backlit Keyboard','HDD Capacity','Battery Life','Fast Charging Support']
x=df[cols].copy()
print(x)
y=df.iloc[:,-1]
print(y)
print(x.head())
print(x.isnull().sum())
print(x['Display Type'].value_counts()/(len(x)-x['Display Type'].isnull().sum())*100)
print(x['Display Features'].value_counts()/(len(x)-x['Display Features'].isnull().sum())*100)
print(x['Display Touchscreen'].value_counts()/(len(x)-x['Display Touchscreen'].isnull().sum())*100)
print(x['Graphic Processor'].value_counts()/(len(x)-x['Graphic Processor'].isnull().sum())*100)
print(x['SSD Capacity'].value_counts()/(len(x)-x['SSD Capacity'].isnull().sum())*100)
print(x['Graphic Processor'].value_counts())
print(x['Capacity'].value_counts()/(len(x)-x['Capacity'].isnull().sum())*100)
print(x['SSD Capacity'].value_counts()/(len(x)-x['SSD Capacity'].isnull().sum())*100)
print(x['Web-cam'].value_counts()/(len(x)-x['Web-cam'].isnull().sum())*100)
print(x['Fingerprint scanner'].value_counts()/(len(x)-x['Fingerprint scanner'].isnull().sum())*100)
print(x['Face Recognition'].value_counts()/(len(x)-x['Face Recognition'].isnull().sum())*100)
print(x['Expandable Memory'].value_counts()/(len(x)-x['Expandable Memory'].isnull().sum())*100)
print(x['Backlit Keyboard'].value_counts()/(len(x)-x['Backlit Keyboard'].isnull().sum())*100)
print(x['HDD Capacity'].value_counts()/(len(x)-x['HDD Capacity'].isnull().sum())*100)
print(x['Battery Life'].value_counts()/(len(x)-x['Battery Life'].isnull().sum())*100)

features1 = [
    'Operating System', 'Display Touchscreen', 'SSD Capacity',
    'Web-cam', 'Fingerprint scanner', 'Face Recognition',
    'Battery Life', 'Fast Charging Support'
]
for feature in features1:
    x.fillna({feature:x[feature].mode()[0]},inplace=True)


features2= [
    'Display Type', 'Display Features', 'Graphic Processor','Expandable Memory', 'Backlit Keyboard', 'HDD Capacity'
]

for feature in features2:
    x.fillna({feature:"None"},inplace=True)

print(x.isnull().sum())

  
print(x['Fast Charging Support'].unique())
print(x['Display Type'].unique())
print(x['Display Type'].value_counts()/(len(x)-x['Display Type'].isnull().sum())*100)