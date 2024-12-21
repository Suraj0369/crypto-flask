from flask import Flask, jsonify, request
import numpy as np
from Prediction import func  
from flask_cors import CORS
#new 
from pymongo import MongoClient



app = Flask(__name__)
CORS(app)

# Initialize MongoDB Client
client = MongoClient('mongodb://localhost:27017/')
db = client['dbtest']  # Replace with your database name
db = client['community_forum']  # Replace with your database name
messages_collection = db['messages']
users_collection = db['users']

#new func
@app.route('/messages', methods=['GET'])
def get_messages():
    messages = list(messages_collection.find({}, {'_id': 0}))  # Exclude _id field
    return jsonify(messages)

@app.route('/messages', methods=['POST'])
def add_message():
    data = request.get_json()
    message = data.get('message')
    messages_collection.insert_one({'message': message})
    return jsonify({'message': 'Message added successfully'}), 201


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')

    # Hash the password

    # Save user to MongoDB
    user = {
        'username': username,
        'password': password,
        'email': email
    }
    users_collection.insert_one(user)

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # Find user in MongoDB
    user = users_collection.find_one({'username': username})
    password = users_collection.find_one({'password': password})
    if user and  password:
        return jsonify({'message': 'Login successful'}), 200
    else:
        return jsonify({'message': 'Invalid credentials'}), 401
    
@app.route('/', methods=['POST', 'GET'])
def index():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        print(symbol)
        # Call the prediction function
        predicted_price = func(symbol)
        
        # return str(predicted_price)
        return jsonify({'predicted_price': str(predicted_price)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host="localhost", port=5000)
