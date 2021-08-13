#import pandas as pd
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

dataset = pd.read_csv('dataset.csv',encoding= 'unicode_escape' )
dataset=dataset.loc[:,['tweet_text','label']]


X = dataset['tweet_text']
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(X)
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

klasifikasiModel = LinearSVC()
klasifikasiModel.fit(X_train, y_train)

y_pred = klasifikasiModel.predict(X_test)

text = input("masukkan text dalam bahasa inggris: ")
output = tfidf.transform([text])
klasifikasiModel.predict(output)
if klasifikasiModel.predict(output) == 1:
    print(text,"======== is positif - sentiment")
else:
    print(text,"======== is negatif - sentiment")