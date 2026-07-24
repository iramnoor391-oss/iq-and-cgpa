#ye chu ml project
#athhe leagzine keh athhe phitrawai
from flask import Flask, render_template, request
import pickle as pkl

with open('mlp_slp_lpa_model.pkl','rb') as f:
    model = pkl.load(f)
with open('scaler.pkl','rb') as f:
    sc = pkl.load(f)

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])

def lpa_predictor():
    if request.method=='POST':
        iq = float(request.form['iq'])
        cgpa = float(request.form['cgpa'])
        iq_scaled = sc.transform([[iq, cgpa]])
        result = model.predict(iq_scaled)[0]
        return render_template('index.html', prediction=f"{result} LPA")
    else:
        return render_template('index.html', prediction=None)
    
if __name__=='__main__':
    app.run(debug=True)