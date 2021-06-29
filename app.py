from flask import Flask, render_template, redirect, url_for, request
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import json
import mysql.connector

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

@app.route("/dataset")
def dataset():

    mydb.connect()
    cursor = mydb.cursor()

    cursor.execute("SELECT * FROM dataset")

    rows = cursor.fetchall()

    cursor.close()
    mydb.close()

    data = []

    for x in rows:
        data.append({
            "tweet":x[0],
            "label":x[1]
        })

    return render_template("Dataset.html",data=json.dumps(data))

@app.route("/importdataset", methods=["POST"])
def importdataset():
    
    dataset = request.files["dataset"]

    document = pd.read_excel(dataset)

    payload = []

    for x in document.iterrows():
        payload.append((x[1]["caption"],x[1]["label"]))


    mydb.connect()
    cursor = mydb.cursor()

    cursor.execute("DELETE FROM dataset")
    cursor.executemany("INSERT INTO dataset VALUES (%s,%s)",payload)

    mydb.commit()
    cursor.close()
    mydb.close()

    return redirect(url_for("dataset"))

if __name__=='__main__':
    app.run(debug=True)