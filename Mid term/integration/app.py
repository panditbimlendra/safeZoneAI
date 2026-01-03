# app.py
from flask import Flask, request, jsonify
import carcrash  # your converted notebook functions

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Process with your car crash model
    result = carcrash.predict_crash(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)