# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pymongo
# import bcrypt

# app = Flask(__name__)
# CORS(app)

# # MongoDB Connection
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["AuthDatabase"]
# users_collection = db["users"]

# @app.route("/signup", methods=["POST"])
# def signup():
#     data = request.json
#     name = data.get("name")
#     email = data.get("email")
#     password = data.get("password")

#     if users_collection.find_one({"email": email}):
#         return jsonify({"success": False, "message": "Email already exists"}), 400

#     hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
#     users_collection.insert_one({"name": name, "email": email, "password": hashed_password})

#     return jsonify({"success": True, "message": "Signup successful"})

# @app.route("/login", methods=["POST"])
# def login():
#     data = request.json
#     email = data.get("email")
#     password = data.get("password")

#     user = users_collection.find_one({"email": email})
#     if not user or not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
#         return jsonify({"success": False, "message": "Invalid credentials"}), 401

#     return jsonify({"success": True, "message": "Login successful", "user": {"name": user["name"], "email": user["email"]}})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Blueprint, request, jsonify
import pymongo
import bcrypt

auth_bp = Blueprint("auth", __name__)

# MongoDB Connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["AuthDatabase"]
users_collection = db["users"]

@auth_bp.route("/signup", methods=["POST"])
def signup():
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if users_collection.find_one({"email": email}):
        return jsonify({"success": False, "message": "Email already exists"}), 400

    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    users_collection.insert_one({"name": name, "email": email, "password": hashed_password})

    return jsonify({"success": True, "message": "Signup successful"})

@auth_bp.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    user = users_collection.find_one({"email": email})
    if not user or not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    return jsonify({"success": True, "message": "Login successful", "user": {"name": user["name"], "email": user["email"]}})
