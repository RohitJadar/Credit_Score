
import numpy as np
from flask import Flask,request,app,render_template
from xgboost import XGBClassifier

def predciton(x):
    XGB=XGBClassifier()
   
   
    XGB.load_model("XGBOOST.json")
    
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
    app.run()
