import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

df = pd.read_csv('Loans_Dataset.csv', sep=',')
df.info() 

df.isnull().sum()

df.head()
df.tail()

X = df.drop(columns=['result'])
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

DTree_model = DecisionTreeClassifier(criterion = "entropy", random_state = 100, max_depth=3, min_samples_leaf=5)

DTree_model.fit(X_train, y_train)

y_pred = DTree_model.predict(X_test)

print('Training score: {:.2f}'.format(DTree_model.score(X_train, y_train)))
print('Test score: {:.2f}'.format(DTree_model.score(X_test, y_test)))

client = np.array([[300, 10000, 300, 4000]])

DTree_model.predict(client)
