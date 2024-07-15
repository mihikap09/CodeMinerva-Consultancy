# app/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from model import load_model, make_recommendations

app = Flask(__name__)
CORS(app)  # Allow Cross-Origin Requests

# Load the model
model = load_model()

@app.route('/api/permission', methods=['POST'])
def permission():
    data = request.get_json()
    permission_granted = data.get('permission_granted')
    if permission_granted:
        # Handle permission granted (e.g., save to database)
        return jsonify({'message': 'Permission granted'}), 200
    else:
        return jsonify({'message': 'Permission denied'}), 400

@app.route('/api/recommendations', methods=['POST'])
def recommendations():
    data = request.get_json()
    user_data = data.get('user_data')
    recommendations = make_recommendations(user_data, model)
    return jsonify({'recommendations': recommendations}), 200

if __name__ == '__main__':
    app.run(debug=True)
