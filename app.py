

import numpy as np
import joblib
import numpy as np
import socket
from flask import Flask,request,app,render_template


def predciton(x):
    scalar=joblib.load("scaler.joblib")


    XGB=joblib.load("XGBOOST.joblib")
    y_pred=XGB.predict(x)




    pred={0:'Good', 1:'Poor', 2:'Standard'}

    return pred[y_pred[0]]




app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

 
    data=np.array([float(x) if isinstance(x,str) else x  for x in request.form.values() ])
  

    predict=predciton([data[6:]])

    return render_template("result.html",var=predict)

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0:5000")
