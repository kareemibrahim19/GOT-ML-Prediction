import pandas as pd
import joblib

df = pd.read_csv(r"C:\Kareem\AIE\Game of the theonesModel\data\character-predictions.csv")

df.info()

df.isnull().sum()

df.dropna(axis=1, thresh=1000, inplace=True)

df.isnull().sum()

df['house'].value_counts()

df.drop('house', axis=1, inplace=True)

df

df.drop('name', axis=1, inplace=True)

df.corr()['isAlive'].sort_values(ascending=False)

df.drop(['S.No', 'book5', 'book4', 'book3', 'book2', 'book1', 'isNoble', 'isMarried'], axis=1, inplace=True)

df.corr()['isAlive'].sort_values(ascending=False)

df

df.describe()

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(),annot=True, cmap='coolwarm')
plt.show()

sns.countplot(x='isAlive', data=df)
plt.show()

sns.histplot(x='popularity', data=df)
plt.show()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df['numDeadRelations'].value_counts()

df['male_alive'] = ((df['male'] == 1) & (df['isAlive'] == 1)).astype(int)
df['female_alive'] = ((df['male'] == 0) & (df['isAlive'] == 1)).astype(int)

df

x = df.drop('isAlive', axis=1)
y = df['isAlive']
joblib.dump(x.columns.tolist(), 'scaler_features.pkl')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=0.95)
x_pca = pca.fit_transform(x_scaled)

print("Original shape:", x.shape)
print("Reduced shape:", x_pca.shape)

joblib.dump(pca, 'got_pca.pkl')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("_LR Accuracy_")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

y_train_pred = rf.predict(x_train)
y_test_pred = rf.predict(x_test)

print("_RF Accuracy_")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)

y_train_pred = svc.predict(x_train)
y_test_pred = svc.predict(x_test)

print("_SVC Accuracy_")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)

y_train_pred = nb.predict(x_train)
y_test_pred = nb.predict(x_test)

print("_NB Accuracy_")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(x_train, y_train)

y_train_pred = xgb.predict(x_train)
y_test_pred = xgb.predict(x_test)

print("_XGB Accuracy_")
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ann = Sequential([
    Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

train_acc = ann.evaluate(x_train, y_train, verbose=0)[1]
test_acc = ann.evaluate(x_test, y_test, verbose=0)[1]

print("_Ann Accuracy_")
print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)
ann.summary()

joblib.dump(rf, 'got_best_model.pkl')
joblib.dump(scaler, 'got_scaler.pkl')