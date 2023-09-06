import numpy as np
from flask import Flask, render_template, request

import pickle

app = Flask(__name__)

model = pickle.load(open("lr_models.pkl", "rb"))


@app.route("/")
def home():
    return render_template("form.html")


@app.route("/predict", methods=["post"])
def predict():
    feature_selection = [float(x) for x in request.form.values()]
    features = [np.array(feature_selection)]
    prediction = model.predict(features)
    return render_template("form.html", prediction_text="the result is {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)

