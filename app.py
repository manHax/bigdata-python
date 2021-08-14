import tkinter as tk
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

dataset = pd.read_csv('stock_data.csv',encoding= 'unicode_escape' )
dataset=dataset.loc[:,['Text','Sentiment']]


X = dataset['Text']
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(X)
y = dataset['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

klasifikasiModel = LinearSVC()
klasifikasiModel.fit(X_train, y_train)

y_pred = klasifikasiModel.predict(X_test)

root= tk.Tk()
root.title('Klasifikasi Sentimen')
root.geometry('600x400+50+50')

sentim= tk.Entry(root,width=200)
sentim.pack()

def clickla():
    test= len(sentim.get())
    if (test>0):
        aa=sentim.get()
        text = aa
        output = tfidf.transform([text])
        klasifikasiModel.predict(output)
        if klasifikasiModel.predict(output) == 1:
            message.config( text=aa+" ======== is positif - sentiment")
        else:
            message.config( text=aa+' ======== is negatif - sentiment')
    else:
        message.config(text="Masukkan text")
btnkla= tk.Button(root, text='Klasifikasi',command=clickla)
btnkla.pack()
message = tk.Label(root, text="")
message.pack()


root.mainloop()
