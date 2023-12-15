from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from .models import Note ,Client
from . import db
import json
import pandas as pd
import numpy as np
import pickle
# df = pd.read_excel("D:\\MP_SIDTA1\\S2\\fouilleDonne\\defaultOfCreditCardClients.xls",header=1 ,skiprows=(0),index_col=0)
data = np.arange(1,23)
# columns =['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_9', 'PAY_8',
#        'PAY_7', 'PAY_6', 'PAY_5', 'PAY_4', 'BILL_AMT9', 'BILL_AMT8',
#        'BILL_AMT7', 'BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'PAY_AMT9',
#        'PAY_AMT8', 'PAY_AMT7', 'PAY_AMT6', 'PAY_AMT5', 'PAY_AMT4' ]

columns =['LIMIT_BAL',  'PAY_9', 'PAY_8',
       'PAY_7',  'PAY_AMT9','PAY_AMT8',
          'PAY_AMT7', 'PAY_AMT6', 'PAY_AMT5',  'PAY_AMT4'  ]

views = Blueprint('views', __name__)

@views.route('/predict',  methods=['GET','POST'])
@login_required
def predcit():
    if request.method == 'POST':
        solvabilter =0
        data = [[request.form.get('LIMIT_BAL'),  request.form.get('PAY_9'),
                 request.form.get('PAY_7') , request.form.get('PAY_AMT9'),
                 request.form.get('PAY_AMT8'), request.form.get('PAY_AMT7'),
                 request.form.get('PAY_AMT6'), request.form.get('PAY_AMT5'),
                 request.form.get('PAY_AMT4'),request.form.get('PAY_8')
                 ]]

        arr = pd.DataFrame(data=data, columns=columns)
         # arr = np.array(data)
        if request.form.get("algorithme")=="dectreclas":
            with open("modelDciTreCl.pk", "rb") as f:
                modelDciTreCl = pickle.load(f)
                solvabiliter = modelDciTreCl.predict(arr)
        elif  request.form.get("algorithme")=="SGDClassifier":
            with open("modelSGDClassifier.pk" ,"rb" ) as f :
                 modelSGDClassifier =pickle.load(f)
                 solvabiliter = modelSGDClassifier.predict(arr)
        elif request.form.get("algorithme")=="Logreg":
            with open("modelLogisRegression.pk" ,"rb" ) as f :
                 modelLogisRegression =pickle.load(f)
                 solvabiliter = modelLogisRegression.predict(arr)
        elif (request.form.get("algorithme") == "navibase"):
            data = [[request.form.get('LIMIT_BAL'),  request.form.get('PAY_9'),  request.form.get('PAY_AMT9'), request.form.get('PAY_AMT8'),
                     request.form.get('PAY_AMT7') ]]

            arr = pd.DataFrame(data=data, columns=columns)
            # arr = np.array(data)
            with open("modelnaivbase.pk", "rb") as f:
                modelnaivbase = pickle.load(f)
                solvabiliter = modelnaivbase.predict(arr)

        newClient = Client(nomprenom=request.form.get("nomPrenom") ,age=request.form.get("AGE") ,solv=solvabiliter[0])
        # db.session.add(newClient)
        # db.session.commit()
        # cliens = Client.query.getAll()
        return render_template("prediction.html" , client=newClient  ,user=current_user)


    else: return "error"

@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    with open("modelDciTreClAccuracy.pk", "rb") as f:
        accTree = pickle.load(f)
    with open("modelSGDClassifierAccuracy.pk", "rb") as f:
         accSGD = pickle.load(f)
    with open("modelLogisRegressionAccuracy.pk", "rb") as f:
        accLR = pickle.load(f)
    with open("modelnaivbaseAccuracy.pk", "rb") as f:
        accNB= pickle.load(f)
    return render_template("homePage.html", user=current_user ,accTree=accTree ,accSGD=accSGD,accLR=accLR ,accNB=accNB)

@views.route('/information', methods=['GET'])
def info():
    return render_template("information.html" ,user=current_user)




 # data = [[request.form.get('LIMIT_BAL'), request.form.get('SEX'), request.form.get('EDUCATION'),
 #                 request.form.get('MARRIAGE')
 #                    , request.form.get('AGE'), request.form.get('PAY_9'), request.form.get('PAY_8'),
 #                 request.form.get('PAY_7'),
 #                 request.form.get('PAY_6'), request.form.get('PAY_5'), request.form.get('PAY_4'),
 #                 request.form.get('BILL_AMT9'),
 #                 request.form.get('BILL_AMT8'), request.form.get('BILL_AMT7'), request.form.get('BILL_AMT6'),
 #                 request.form.get('BILL_AMT5'),
 #                 request.form.get('BILL_AMT4'), request.form.get('PAY_AMT9'), request.form.get('PAY_AMT8'),
 #                 request.form.get('PAY_AMT7'),
 #                 request.form.get('PAY_AMT6'), request.form.get('PAY_AMT5'), request.form.get('PAY_AMT4')]]