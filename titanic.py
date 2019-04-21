from google.colab import drive
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

drive.mount('/content/gdrive')
bd = pd.read_csv('gdrive/My Drive/titanic.csv') #загрузка датасета

bd = bd.drop(columns=['Name'])
bd = bd.drop(columns=['Ticket'])
bd = bd.drop(columns=['Fare'])  #удаление всех неважных столбцов

bd['Sex'] = pd.get_dummies(bd['Sex'])
bd['Cabin'] = pd.get_dummies(bd['Cabin'])
bd['Embarked'] = pd.get_dummies(bd['Embarked'])  #приведение в читаемый вид важных данных

X = bd.drop(columns=['Survived'])
y = bd['Survived']  #разделение данных на ответы и сами данные

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  #выборка тестовых данных и контрольных

mis_replacer = preprocessing.Imputer(strategy="mean")
X_train = pd.DataFrame(data=mis_replacer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(data=mis_replacer.fit_transform(X_test), columns=X_train.columns)  #убираем все Nan

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)  #обучение модели и предсказания

print(accuracy_score(y_test, y_pred))  #вывод значения точности