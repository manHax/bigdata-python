from tkinter import *
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

dataset = pd.read_csv('stock_data.csv',encoding= 'unicode_escape' )
dataset=dataset.loc[:,['tweets','sentiment']]


X = dataset['tweets']
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(X)
y = dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 12)

klasifikasiModel = LinearSVC()
klasifikasiModel.fit(X_train, y_train)

#y_pred = klasifikasiModel.predict(X_test)


def clearAll() : 
    overallField.delete(0, END)
    textArea.delete(1.0, END)
    
def detect_sentiment():
    sentence = textArea.get("1.0", "end")
    output = tfidf.transform([sentence])
    if klasifikasiModel.predict(output) == 1 :
        string = "Positive"
    else:
        string = "Negative"
    overallField.insert(10, string)


if __name__ == "__main__" :
    gui = Tk()
    gui.config(background =  "light green")
    gui.title("Deteksi Sentimen")
    gui.geometry("900x500")
    
    enterText = Label(gui, text = "Masukkan kalimat dalam bahasa inggris",bg = "light green")
    textArea = Text(gui, height = 10, width = 97, font = "lucida 13")
    check = Button(gui, text = "Check Sentiment", fg = "Black", bg = "white", activebackground="red", command = detect_sentiment)
    overall = Label(gui, text = "Hasil: ", bg = "light green")
    overallField = Entry(gui)
    clear = Button(gui, text = "Clear", fg = "Black", bg = "Red", command = clearAll)

    enterText.grid(row = 0, column = 2)
    #enterText2.grid(row = 1, column = 2)
    textArea.grid(row = 2, column = 2, padx = 10, sticky = W)
    check.grid(row = 3, column = 2, padx=2, pady=12)
    overall.grid(row = 9, column = 2, padx=2, pady=2)
    overallField.grid(row = 10, column = 2)
    clear.grid(row = 11, column = 2, padx=2, pady=20)

    # start the GUI
    gui.mainloop()
