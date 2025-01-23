from flask import Flask
from flask import render_template,request,session
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
app=Flask(__name__)
app.secret_key='akvsmlbebpomprempnm'
if os.path.exists('temp.csv')==True:
    os.remove('temp.csv')
if os.path.exists('model.pickle')==True:
    os.remove('model.pickle')
@app.route('/Upload',methods=['POST'])
def upload():
    if request.method=='POST':
        file=request.files['file']
        file.save("temp.csv")
        data=pd.read_csv('temp.csv')
        data.dropna(inplace=True)
        encoder=LabelEncoder()
        target=[col for col in data.columns if 'downtime' in col.lower()][0]
        session['target_name']=target
        data[target]=encoder.fit_transform(data[target])
        data.to_csv('temp.csv',index=False)
        with open('enc.pickle','wb') as f:
            pickle.dump(encoder,f)
        return "Uploaded Sucessfully"
@app.route('/Train',methods=['POST'])
def train():
    if request.method=='POST':
        if os.path.exists('temp.csv')==False:
            return 'Please Upload the file'
        else:
            data=pd.read_csv('temp.csv')
            id=[col for col in data.columns if 'machine' in col.lower()][0]
            x_train,x_test,y_train,y_test=train_test_split(data.drop(columns=[session['target_name'],id]),data[session['target_name']],test_size=0.2)
            model=LogisticRegression()
            print(x_train.columns)
            model.fit(x_train,y_train)
            with open('model.pickle','wb') as f:
                pickle.dump(model,f)
            y_pred=model.predict(x_test)
            return {'ACCURACY':accuracy_score(y_test,y_pred),'F1_SCORE':f1_score(y_test,y_pred),'RECALL':recall_score(y_test,y_pred),'PRECISION':precision_score(y_test,y_pred)}
        
@app.route('/predict',methods=['POST'])
def test():
    if request.method=='POST':
        if os.path.exists('model.pickle')==False:
            return 'Please Train the model first'
        else:
            with open('model.pickle','rb') as f:
                model=pickle.load(f)
            with open('enc.pickle','rb') as f:
                enc=pickle.load(f)
            data=request.get_json()
            
            values=list(data.values())
            print(values)
            result=model.predict([[values[0],values[1]]])

            return {'downtime':str(enc.inverse_transform(result)[0]),'confidence':str(model.predict_proba([[values[0],values[1]]])[0][result[0]])}       
            


        




if __name__=='__main__':
    app.run(debug=True)