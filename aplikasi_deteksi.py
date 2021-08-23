
from tkinter import *
#import numpy as np
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

y_pred = klasifikasiModel.predict(X_test)


def clearAll() : 
    # deleting the content from the entry box
    overallField.delete(0, END)
    # whole content of text area  is deleted
    textArea.delete(1.0, END)
    
def detect_sentiment():
    # get a whole input content from text box
    sentence = textArea.get("1.0", "end")
    output = tfidf.transform([sentence])
 
  # decide sentiment as positive, negative and neutral
    if klasifikasiModel.predict(output) == 1 :
        string = "Positive"
    else:
        string = "Negative"

    overallField.insert(10, string)

# Driver Code
if __name__ == "__main__" :
    # Create a GUI window
    gui = Tk()
     
    # Set the background colour of GUI window
    gui.config(background =  "light green")
 
    # set the name of tkinter GUI window
    gui.title("Deteksi Sentimen")
 
    # Set the configuration of GUI window
    gui.geometry("900x500")
 
    # create a label : Enter Your Task
    enterText = Label(gui, text = "Masukkan kalimat dalam bahasa inggris",
                                     bg = "light green")
    enterText2 = Label(gui, text = "Keakuratan deteksi: ",
                                     bg = "light green")
 
    # create a text area for the root
    # with lunida 13 font
    # text area is for writing the content
    textArea = Text(gui, height = 10, width = 97, font = "lucida 13")
 
    # create a Submit Button and place into the root window
    # when user press the button, the command or 
    # function affiliated to that button is executed 
    check = Button(gui, text = "Check Sentiment", fg = "Black",
                         bg = "white", activebackground="red", command = detect_sentiment)
 
    # Create a overall : label
    overall = Label(gui, text = "Hasil: ", bg = "light green")
 
    # create a text entry box 
    overallField = Entry(gui)
 
    # create a Clear Button and place into the root window
    # when user press the button, the command or 
    # function affiliated to that button is executed .
    clear = Button(gui, text = "Clear", fg = "Black",
                      bg = "Red", command = clearAll)
     

    enterText.grid(row = 0, column = 2)
    enterText2.grid(row = 1, column = 2)
     
    textArea.grid(row = 2, column = 2, padx = 10, sticky = W)
     
    check.grid(row = 3, column = 2, padx=2, pady=12)
     
    overall.grid(row = 9, column = 2, padx=2, pady=2)

    overallField.grid(row = 10, column = 2)
     
    clear.grid(row = 11, column = 2, padx=2, pady=20)

    # start the GUI
    gui.mainloop()
