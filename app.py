from flask import Flask,jsonify,request
from classifier import getPrediction

app=Flask(__name__)

@app.route('/predict-alphabet',methods=['POST'])
def predict_digit():
    image=request.files.get('digit')
    result=getPrediction(image)
    return jsonify({
        'prediction':result
    }),200

if(__name__=="__main__"):
    app.run(debug=True)