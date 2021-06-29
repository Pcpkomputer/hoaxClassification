from flask import Flask, render_template
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("Dashboard.html")

@app.route("/dataset")
def dataset():
    return render_template("Dataset.html")

if __name__=='__main__':
    app.run(debug=True)