import os
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('titanic.csv')
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

df_selected = df.loc[:, ['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]

df_selected = df_selected.dropna()

x = df_selected.drop('Survived', axis=1)
y = df_selected['Survived']

clf = DecisionTreeClassifier()
clf.fit(x, y)

feature_importances = clf.feature_importances_
feature_importances_df = pd.DataFrame({'Feature': x.columns, 'Importance': feature_importances})

print(feature_importances_df)
