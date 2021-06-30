from flask import Flask, render_template, redirect, url_for, request
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import json
import mysql.connector
from sklearn.metrics import confusion_matrix
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import classification_report,accuracy_score


app = Flask(__name__)

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="hoaxnewsclassification"
)

@app.route("/")
def index():
    return render_template("Dashboard.html")

@app.route("/preprocessing",methods=["POST","GET"])
def preprocessing():
    if request.method=="POST":
        
        mydb.connect()
        cursor = mydb.cursor()

        cursor.execute("SELECT * FROM dataset")
        
        rows = cursor.fetchall()

        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()

        cursor.execute("DELETE FROM preprocessing")

        payload = []
        
        for index,x in enumerate(rows):
            #### MELAKUKAN PROSES STEMMING STOPWORD BAHASA INDONESIA
            satu = stopword.remove(x[1])
        
            #### MENGHILANGKAN TEXT TIDAK PENTING SEPERTI HASHTAG DAN MENTION
            dua = re.sub(r"@[^\s]+"," ",satu)
            dua = re.sub(r"#[^\s]+"," ",dua)
            dua = re.sub(r"\."," ",dua)
            dua = re.sub(r"http[^\s]+"," ",dua)
            dua = re.sub(r"\?"," ",dua)
            dua = re.sub(r","," ",dua)
            dua = re.sub(r"”"," ",dua)
            dua = re.sub(r"co/[^\s]+"," ",dua)
            dua = re.sub(r":'\)"," ",dua)
            dua = re.sub(r":\)","",dua)
            dua = re.sub(r"&"," ",dua)
            dua = re.sub(r'\"([^\"]+)\"',"\g<1>",dua)
            dua = re.sub(r'\([^\)]+\"',"",dua)
            dua = re.sub(r'\((.+)\)',"\g<1>",dua)
            dua = re.sub(r'-'," ",dua)
            dua = re.sub(r':\('," ",dua)
            dua = re.sub(r':'," ",dua)
            dua = re.sub(r'\('," ",dua)
            dua = re.sub(r'\)'," ",dua)
            dua = re.sub(r"'"," ",dua)
            dua = re.sub(r'"'," ",dua)
            dua = re.sub(r';'," ",dua)
            dua = re.sub(r':v'," ",dua)
            dua = re.sub(r'²'," ",dua)
            dua = re.sub(r':"\)'," ",dua)
            dua = re.sub(r'\[\]'," ",dua)
            dua = re.sub(r'“',"",dua)
            dua = re.sub(r'_'," ",dua)
            dua = re.sub(r'—'," ",dua)
            dua = re.sub(r'…'," ",dua)
            dua = re.sub(r'='," ",dua)
            dua = re.sub(r'\/'," ",dua)
            dua = re.sub(r'\[\w+\]'," ",dua)
            dua = re.sub(r'!'," ",dua)
            dua = re.sub(r"'"," ",dua)
            dua = re.sub(r'\s+'," ",dua)
            dua = re.sub(r'^RT',"",dua) 
            dua = re.sub(r'\s+$',"",dua)   
            dua = re.sub(r'^\s+',"",dua)   
            #### MENGUBAH CASE KATA MENJADI LOWERCASE
            tiga = dua.lower()
            #### MENGUBAH KATA KEKINIAN MENJADI SESUAI PUEBI
            payload.append((index+1,x[1],tiga,x[2]))

        cursor.executemany("INSERT INTO preprocessing VALUES (%s,%s,%s,%s)",payload)

        mydb.commit()
        cursor.close()
        mydb.close()

        return redirect(url_for("preprocessing"))

    mydb.connect()
    cursor = mydb.cursor()

    cursor.execute("SELECT * FROM preprocessing")
    rows = cursor.fetchall()

    payload = []

    for index,x in enumerate(rows):
        payload.append({
            "no":x[0],
            "tweetsebelum":x[1],
            "tweetsesudah":x[2],
            "label":x[3]
        })

    cursor.close()
    mydb.close()

    return render_template("Preprocessing.html",data=json.dumps(payload))

@app.route("/dataset")
def dataset():

    mydb.connect()
    cursor = mydb.cursor()

    cursor.execute("SELECT * FROM dataset")

    rows = cursor.fetchall()

    cursor.close()
    mydb.close()

    data = []

    for index,x in enumerate(rows):
        data.append({
            "id":index+1,
            "tweet":x[1],
            "label":x[2]
        })

    return render_template("Dataset.html",data=json.dumps(data))

@app.route("/pengujian", methods=["POST","GET"])
def pengujian():
    
    if request.method=="POST":
        mydb.connect()
        cursor = mydb.cursor()

        cursor.execute("SELECT * FROM preprocessing")

        rows = cursor.fetchall()

        corpus = [x[2] for x in rows]
        corpus = np.array(corpus)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        y = [x[3] for x in rows]
        y = np.array(y)

        kf = KFold(n_splits=4)
        kf.get_n_splits(X)

        payload = []
        payload_confusion = []
        payload_akurasi = []

        for train_index, test_index in kf.split(X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_test_raw = corpus[test_index]

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = MultinomialNB()
            clf.fit(X_train,y_train)

            predicted = clf.predict(X_test)

            corp = X_test_raw
            pred = predicted

            confusion = confusion_matrix(y_test,pred, labels=["hoax","valid"])
            akurasi = accuracy_score(y_test,pred)

            payload_akurasi.append(str(int(akurasi*100))+"%")

            tn, fp, fn, tp = confusion.ravel()

            payload_confusion.append({
                "tn":tn,
                "fp":fp,
                "fn":fn,
                "tp":tp
            })

            zipped = list(zip(corp,pred,y_test))

            p = []

            for l in zipped:
                p.append({
                    "corpus":l[0],
                    "label":l[1],
                    "actual":l[2]
                })
            payload.append(p)

        print(payload_akurasi)

        cursor.close()
        mydb.close()

        return render_template("Pengujian.html",show=True,confusion=list(enumerate(payload_confusion)),akurasi=payload_akurasi,data=list(enumerate(payload)))

    return render_template("Pengujian.html")


@app.route("/importdataset", methods=["POST"])
def importdataset():
    
    dataset = request.files["dataset"]

    document = pd.read_excel(dataset)

    payload = []

    counter = 0

    for x in document.iterrows():
        counter=counter+1
        payload.append((counter,x[1]["caption"],x[1]["label"]))


    mydb.connect()
    cursor = mydb.cursor()

    cursor.execute("DELETE FROM dataset")
    cursor.executemany("INSERT INTO dataset VALUES (%s,%s,%s)",payload)

    mydb.commit()
    cursor.close()
    mydb.close()

    return redirect(url_for("dataset"))


if __name__=='__main__':
    app.run(debug=True)