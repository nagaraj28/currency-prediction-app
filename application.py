from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

application = Flask(__name__)
CORS(application)


# Define a function to load a trained model for a given currency
def load_model(currency):
    model = pickle.load(open(f'{currency}_model.pkl', 'rb'))
    return model

# Define a predict function for the API
@application.route('/predict', methods=['POST'])
def predict():
    # # Get the currency code and input data from the request
    # currency = request.form['currency']
    # open_rate = float(request.form['open_rate'])
    # high_rate = float(request.form['high_rate'])
    # low_rate = float(request.form['low_rate'])
    input_data = request.get_json()
    currency = input_data['currency']
    open_rate = input_data['open_rate']
    high_rate = input_data['high_rate']
    low_rate = input_data['low_rate']
    # Load the model for the currency
    model = load_model(currency)
    # Use the model to predict the closing rate
    prediction = model.predict([[open_rate, high_rate, low_rate]])[0]
    print(prediction)
    # Return the prediction as a JSON response
    return jsonify({'predicted_close_rate': prediction})
@application.route('/predictall', methods=['POST'])
def predictall():
    # # Get the currency code and input data from the request
    # currency = request.form['currency']
    # open_rate = float(request.form['open_rate'])
    # high_rate = float(request.form['high_rate'])
    # low_rate = float(request.form['low_rate'])
    currenciesList = ["inr","cny","jpy","krw","kzt","lkr","mvr","myr","npr","php","pkr","sgd","thb","twd","vnd"]
    input_data = request.get_json()
    currency = input_data['currency']
    open_rate = input_data['open_rate']
    high_rate = input_data['high_rate']
    low_rate = input_data['low_rate']
    resultPredict = {}
    for currencyItem in currenciesList:
        # Load the model for the currency
        model = load_model(currencyItem)
        # Use the model to predict the closing rate
        prediction = model.predict([[open_rate, high_rate, low_rate]])[0]
        resultPredict[currencyItem] = prediction
    # Return the prediction as a JSON response
    return jsonify({'predicted_close_rate': resultPredict})

if __name__ == '__main__':
    application.run(debug=True)
