# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import os
# from PIL import Image

# app = Flask(__name__)
# CORS(app)  # Allow frontend to communicate with backend

# # Load the trained model
# MODEL_PATH = "disaster_segmentation_model.h5"
# model = tf.keras.models.load_model(MODEL_PATH)

# # Function to preprocess image
# def preprocess_image(img_path):
#     img = load_img(img_path, target_size=(256, 256))
#     img = img_to_array(img) / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
#     return img

# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files["file"]
#     img_path = os.path.join("uploads", file.filename)
#     file.save(img_path)
    
#     # Preprocess and predict
#     img = preprocess_image(img_path)
#     pred_mask = model.predict(img)[0]
#     pred_mask = (pred_mask.squeeze() * 255).astype(np.uint8)
    
#     # Convert prediction to PIL image and save
#     pred_image = Image.fromarray(pred_mask)
#     pred_path = os.path.join("uploads", "predicted_" + file.filename)
#     pred_image.save(pred_path)
    
#     return jsonify({"message": "Prediction complete", "predicted_image": pred_path})

# if __name__ == "__main__":
#     os.makedirs("uploads", exist_ok=True)
#     app.run(debug=True)

# from flask import Flask, request, jsonify, send_from_directory  # ✅ Import send_from_directory

# from flask_cors import CORS
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# import os
# from PIL import Image
# import shutil  # To copy files properly

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # ✅ Allow React frontend (port 3000)

# UPLOAD_FOLDER = "uploads"
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     # Save the uploaded file
#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(file_path)

#     # Create a simulated mask image (replace with AI model output)
#     mask_filename = f"mask_{file.filename}"
#     mask_path = os.path.join(app.config["UPLOAD_FOLDER"], mask_filename)

#     shutil.copy(file_path, mask_path)  # Simulate mask generation by copying original

#     # Debugging logs
#     print(f"✅ File saved: {file_path}")
#     print(f"✅ Mask saved: {mask_path}")

#     # Ensure the mask exists before returning the URL
#     if not os.path.exists(mask_path):
#         return jsonify({"error": "Mask generation failed"}), 500

#     # Generate the correct mask URL
#     mask_url = f"http://127.0.0.1:5000/uploads/{mask_filename}"
#     print(f"✅ Mask URL: {mask_url}")  # Debugging

#     response = jsonify({"mask_url": mask_url})
#     response.headers.add("Access-Control-Allow-Origin", "*")
#     return response

# # Route to serve uploaded images
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
# import cv2
# import uuid  # For generating unique filenames

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Adjusted for frontend

# UPLOAD_FOLDER = "uploads"
# MASK_FOLDER = "masks"

# # Create folders if they don't exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(MASK_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["MASK_FOLDER"] = MASK_FOLDER

# # Load the trained model
# MODEL_PATH = "disaster_segmentation_model.h5"
# model = load_model(MODEL_PATH)
# IMG_SIZE = (256, 256)  # Model input size

# def generate_mask(image_path):
#     """ Preprocesses the uploaded image, passes it through the model, and returns the mask path. """
#     img = load_img(image_path, target_size=IMG_SIZE)
#     img_array = img_to_array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Predict mask
#     predicted_mask = model.predict(img_array)[0]  # Get first image mask
#     predicted_mask = (predicted_mask * 255).astype(np.uint8)  # Convert to uint8

#     # Generate unique filename for mask
#     mask_filename = f"mask_{uuid.uuid4().hex}.png"
#     mask_path = os.path.join(app.config["MASK_FOLDER"], mask_filename)

#     # Save mask as image
#     cv2.imwrite(mask_path, predicted_mask)

#     return mask_path

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400

#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     # Generate unique filename
#     unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
#     file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
#     file.save(file_path)

#     try:
#         # Generate actual mask using AI model
#         mask_path = generate_mask(file_path)
#         mask_url = f"http://127.0.0.1:5000/masks/{os.path.basename(mask_path)}"
#         return jsonify({"mask_url": mask_url})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/masks/<filename>')
# def get_mask(filename):
#     return send_from_directory(app.config["MASK_FOLDER"], filename)

# if __name__ == '__main__':
#     app.run(debug=True)




# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
# import cv2

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# UPLOAD_FOLDER = "uploads"
# MASK_FOLDER = "masks"

# # Ensure folders exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(MASK_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["MASK_FOLDER"] = MASK_FOLDER

# # Load the trained model
# MODEL_PATH = "disaster_segmentation_model.h5"
# model = load_model(MODEL_PATH)
# IMG_SIZE = (256, 256)  # Model input size

# def generate_mask(image_path):
#     """ Preprocesses the uploaded image, passes it through the model, and returns the mask. """
#     img = load_img(image_path, target_size=IMG_SIZE)
#     img_array = img_to_array(img) / 255.0  # Normalize
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Predict mask
#     predicted_mask = model.predict(img_array)[0]  # Get first image mask

#     # Convert mask to uint8 (0-255)
#     predicted_mask = (predicted_mask * 255).astype(np.uint8)

#     # Ensure mask has 3 channels (convert grayscale to RGB)
#     if len(predicted_mask.shape) == 2:  # If grayscale (H, W)
#         predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2RGB)

#     mask_filename = f"mask_{os.path.splitext(os.path.basename(image_path))[0]}.png"  # Ensure PNG format
#     mask_path = os.path.join(app.config["MASK_FOLDER"], mask_filename)

#     # Save mask as image
#     success = cv2.imwrite(mask_path, predicted_mask)
#     if not success:
#         raise RuntimeError(f"Failed to write mask image: {mask_path}")

#     return mask_path


# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'files[]' not in request.files:
#         return jsonify({"error": "No files uploaded"}), 400

#     files = request.files.getlist('files[]')  # Get multiple files
#     mask_urls = []

#     for file in files:
#         if file.filename == '':
#             continue

#         file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#         file.save(file_path)

#         # Generate actual mask using AI model
#         mask_path = generate_mask(file_path)
#         mask_url = f"http://127.0.0.1:5000/masks/{os.path.basename(mask_path)}"
#         mask_urls.append(mask_url)

#     return jsonify({"mask_urls": mask_urls})

# @app.route('/masks/<filename>')
# def get_mask(filename):
#     return send_from_directory(app.config["MASK_FOLDER"], filename)

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
# import cv2

# app = Flask(__name__)
# CORS(app)  # Allow all origins for now

# UPLOAD_FOLDER = "uploads"
# MASK_FOLDER = "masks"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(MASK_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["MASK_FOLDER"] = MASK_FOLDER

# # Load the model
# MODEL_PATH = "disaster_multi_output_model.keras"

# try:
#     model = load_model(MODEL_PATH)
#     print("[INFO] Model loaded successfully.")
# except Exception as e:
#     print(f"[ERROR] Failed to load model: {e}")
#     model = None  # Handle failure

# IMG_SIZE = (256, 256)
# disaster_classes = ["Deforestation", "Landslide", "Flood"]

# def process_image(image_path):
#     try:
#         img = load_img(image_path, target_size=IMG_SIZE)
#         img_array = img_to_array(img) / 255.0  # Normalize
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         predictions = model.predict(img_array)

#         predicted_mask = predictions[0]
#         predicted_class_index = np.argmax(predictions[1])

#         predicted_mask = (predicted_mask.squeeze() * 255).astype(np.uint8)
#         if len(predicted_mask.shape) == 2:  # Convert grayscale to RGB
#             predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2RGB)

#         # Save the mask
#         mask_filename = f"mask_{os.path.basename(image_path)}"
#         mask_path = os.path.join(app.config["MASK_FOLDER"], mask_filename)
#         cv2.imwrite(mask_path, predicted_mask)

#         predicted_disaster = disaster_classes[predicted_class_index]

#         return mask_path, predicted_disaster
#     except Exception as e:
#         print(f"[ERROR] Image processing failed: {e}")
#         return None, None

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'files[]' not in request.files:
#         return jsonify({"error": "No files uploaded"}), 400

#     files = request.files.getlist('files[]')
#     results = []

#     for file in files:
#         if file.filename == '':
#             continue

#         file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#         file.save(file_path)

#         mask_path, predicted_disaster = process_image(file_path)
#         if mask_path and predicted_disaster:
#             results.append({
#                 "original_url": f"http://127.0.0.1:5000/uploads/{file.filename}",
#                 "mask_url": f"http://127.0.0.1:5000/masks/{os.path.basename(mask_path)}",
#                 "disaster_class": predicted_disaster
#             })
#         else:
#             return jsonify({"error": "Processing failed for some images"}), 500

#     return jsonify({"results": results})

# @app.route('/uploads/<filename>')
# def get_original(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# @app.route('/masks/<filename>')
# def get_mask(filename):
#     return send_from_directory(app.config["MASK_FOLDER"], filename)

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
# import cv2

# app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

# UPLOAD_FOLDER = "uploads"
# MASK_FOLDER = "masks"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(MASK_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["MASK_FOLDER"] = MASK_FOLDER

# # Load the model
# MODEL_PATH = "disaster_multi_output_model.keras"

# try:
#     model = load_model(MODEL_PATH)
#     print("[INFO] Model loaded successfully.")
# except Exception as e:
#     print(f"[ERROR] Failed to load model: {e}")
#     model = None  # Handle failure

# IMG_SIZE = (256, 256)
# disaster_classes = ["Deforestation", "Landslide", "Flood"]

# # Mitigation Strategies for Each Disaster
# mitigation_measures = {
#     "Deforestation": [
#         "Promote afforestation and reforestation.",
#         "Implement sustainable logging practices.",
#         "Strengthen forest protection laws.",
#         "Encourage agroforestry techniques.",
#         "Improve land-use planning.",
#         "Ban illegal logging activities.",
#         "Raise awareness about conservation.",
#         "Reduce paper and wood consumption.",
#         "Implement carbon offset programs.",
#         "Support indigenous forest management."
#     ],
#     "Landslide": [
#         "Plant deep-rooted vegetation.",
#         "Construct retaining walls and terraces.",
#         "Improve drainage systems.",
#         "Implement early warning systems.",
#         "Enforce strict land-use policies.",
#         "Avoid deforestation in high-risk areas.",
#         "Use slope stabilization techniques.",
#         "Develop hazard maps for planning.",
#         "Monitor soil movement using sensors.",
#         "Train communities on emergency response."
#     ],
#     "Flood": [
#         "Improve urban drainage systems.",
#         "Construct flood barriers and levees.",
#         "Promote rainwater harvesting.",
#         "Implement sustainable land management.",
#         "Enforce floodplain zoning regulations.",
#         "Strengthen embankments along rivers.",
#         "Develop community flood response plans.",
#         "Use permeable surfaces in urban areas.",
#         "Implement reforestation in watersheds.",
#         "Deploy real-time flood monitoring systems."
#     ]
# }

# def process_image(image_path):
#     try:
#         img = load_img(image_path, target_size=IMG_SIZE)
#         img_array = img_to_array(img) / 255.0  # Normalize
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         predictions = model.predict(img_array)

#         predicted_mask = predictions[0]
#         predicted_class_index = np.argmax(predictions[1])
#         confidence_score = np.max(predictions[1]) * 100  # Convert to percentage

#         predicted_mask = (predicted_mask.squeeze() * 255).astype(np.uint8)
#         if len(predicted_mask.shape) == 2:  # Convert grayscale to RGB
#             predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2RGB)

#         # Save the mask
#         mask_filename = f"mask_{os.path.basename(image_path)}"
#         mask_path = os.path.join(app.config["MASK_FOLDER"], mask_filename)
#         cv2.imwrite(mask_path, predicted_mask)

#         predicted_disaster = disaster_classes[predicted_class_index]

#         return mask_path, predicted_disaster, confidence_score
#     except Exception as e:
#         print(f"[ERROR] Image processing failed: {e}")
#         return None, None, None


# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'files[]' not in request.files:
#         return jsonify({"error": "No files uploaded"}), 400

#     files = request.files.getlist('files[]')
#     results = []

#     for file in files:
#         if file.filename == '':
#             continue

#         file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#         file.save(file_path)

#         mask_path, predicted_disaster, confidence_score = process_image(file_path)
#         if mask_path and predicted_disaster:
#             results.append({
#                 "original_url": f"http://127.0.0.1:5000/uploads/{file.filename}",
#                 "mask_url": f"http://127.0.0.1:5000/masks/{os.path.basename(mask_path)}",
#                 "disaster_class": predicted_disaster,
#                 "accuracy": confidence_score  # Add confidence score
#             })
#         else:
#             return jsonify({"error": "Processing failed for some images"}), 500

#     return jsonify({"results": results})


# @app.route('/uploads/<filename>')
# def get_original(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# @app.route('/masks/<filename>')
# def get_mask(filename):
#     return send_from_directory(app.config["MASK_FOLDER"], filename)

# if __name__ == '__main__':
#     app.run(debug=True)





# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
# import cv2
# from auth import auth_bp 
# from flask_session import Session# Import authentication routes

# app = Flask(__name__)

# # Secret key for session management
# app.secret_key = "your_secret_key"

# # Configure session
# app.config["SESSION_TYPE"] = "filesystem"
# app.config["SESSION_PERMANENT"] = False
# app.config["SESSION_USE_SIGNER"] = True
# app.config["SESSION_COOKIE_HTTPONLY"] = True
# app.config["SESSION_COOKIE_SAMESITE"] = "None"
# app.config["SESSION_COOKIE_SECURE"] = False  # Set to True for HTTPS

# # Enable CORS for all routes with credentials support
# CORS(app, supports_credentials=True, origins=["http://localhost:3000"])

# # Initialize session
# Session(app)

# # Register authentication blueprint
# app.register_blueprint(auth_bp, url_prefix="/api/auth")

# UPLOAD_FOLDER = "uploads"
# MASK_FOLDER = "masks"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(MASK_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["MASK_FOLDER"] = MASK_FOLDER

# # Load the model
# MODEL_PATH = "disaster_multi_output_model.keras"

# try:
#     model = load_model(MODEL_PATH)
#     print("[INFO] Model loaded successfully.")
# except Exception as e:
#     print(f"[ERROR] Failed to load model: {e}")
#     model = None  # Handle failure

# IMG_SIZE = (256, 256)
# disaster_classes = ["Deforestation", "Landslide", "Flood"]

# # Mitigation Strategies for Each Disaster
# mitigation_measures = {
#     "Deforestation": [
#         "Promote afforestation and reforestation.",
#         "Implement sustainable logging practices.",
#         "Strengthen forest protection laws.",
#         "Encourage agroforestry techniques.",
#         "Improve land-use planning.",
#         "Ban illegal logging activities.",
#         "Raise awareness about conservation.",
#         "Reduce paper and wood consumption.",
#         "Implement carbon offset programs.",
#         "Support indigenous forest management."
#     ],
#     "Landslide": [
#         "Plant deep-rooted vegetation.",
#         "Construct retaining walls and terraces.",
#         "Improve drainage systems.",
#         "Implement early warning systems.",
#         "Enforce strict land-use policies.",
#         "Avoid deforestation in high-risk areas.",
#         "Use slope stabilization techniques.",
#         "Develop hazard maps for planning.",
#         "Monitor soil movement using sensors.",
#         "Train communities on emergency response."
#     ],
#     "Flood": [
#         "Improve urban drainage systems.",
#         "Construct flood barriers and levees.",
#         "Promote rainwater harvesting.",
#         "Implement sustainable land management.",
#         "Enforce floodplain zoning regulations.",
#         "Strengthen embankments along rivers.",
#         "Develop community flood response plans.",
#         "Use permeable surfaces in urban areas.",
#         "Implement reforestation in watersheds.",
#         "Deploy real-time flood monitoring systems."
#     ]
# }

# def process_image(image_path):
#     try:
#         img = load_img(image_path, target_size=IMG_SIZE)
#         img_array = img_to_array(img) / 255.0  # Normalize
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         predictions = model.predict(img_array)

#         predicted_mask = predictions[0]
#         predicted_class_index = np.argmax(predictions[1])
#         confidence_score = np.max(predictions[1]) * 100  # Convert to percentage

#         predicted_mask = (predicted_mask.squeeze() * 255).astype(np.uint8)
#         if len(predicted_mask.shape) == 2:  # Convert grayscale to RGB
#             predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2RGB)

#         # Save the mask
#         mask_filename = f"mask_{os.path.basename(image_path)}"
#         mask_path = os.path.join(app.config["MASK_FOLDER"], mask_filename)
#         cv2.imwrite(mask_path, predicted_mask)

#         predicted_disaster = disaster_classes[predicted_class_index]

#         return mask_path, predicted_disaster, confidence_score
#     except Exception as e:
#         print(f"[ERROR] Image processing failed: {e}")
#         return None, None, None


# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'files[]' not in request.files:
#         return jsonify({"error": "No files uploaded"}), 400

#     files = request.files.getlist('files[]')
#     results = []

#     for file in files:
#         if file.filename == '':
#             continue

#         file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#         file.save(file_path)

#         mask_path, predicted_disaster, confidence_score = process_image(file_path)
#         if mask_path and predicted_disaster:
#             results.append({
#                 "original_url": f"http://127.0.0.1:5000/uploads/{file.filename}",
#                 "mask_url": f"http://127.0.0.1:5000/masks/{os.path.basename(mask_path)}",
#                 "disaster_class": predicted_disaster,
#                 "accuracy": confidence_score  # Add confidence score
#             })
#         else:
#             return jsonify({"error": "Processing failed for some images"}), 500

#     return jsonify({"results": results})


# @app.route('/uploads/<filename>')
# def get_original(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# @app.route('/masks/<filename>')
# def get_mask(filename):
#     return send_from_directory(app.config["MASK_FOLDER"], filename)

# if __name__ == '__main__':
#     app.run(debug=True)



# from flask import Flask, request, jsonify, send_from_directory
# from flask_cors import CORS
# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
# import cv2
# from auth import auth_bp  # Import authentication routes

# app = Flask(__name__)
# CORS(app)

# # Register authentication routes
# app.register_blueprint(auth_bp)

# UPLOAD_FOLDER = "uploads"
# MASK_FOLDER = "masks"

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(MASK_FOLDER, exist_ok=True)

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["MASK_FOLDER"] = MASK_FOLDER

# # Load the model
# MODEL_PATH = "disaster_multi_output_model.keras"

# try:
#     model = load_model(MODEL_PATH)
#     print("[INFO] Model loaded successfully.")
# except Exception as e:
#     print(f"[ERROR] Failed to load model: {e}")
#     model = None  # Handle failure

# IMG_SIZE = (256, 256)
# disaster_classes = ["Deforestation", "Landslide", "Flood"]

# # Mitigation Strategies for Each Disaster
# mitigation_measures = {
#     "Deforestation": [
#         "Promote afforestation and reforestation.",
#         "Implement sustainable logging practices.",
#         "Strengthen forest protection laws.",
#         "Encourage agroforestry techniques.",
#         "Improve land-use planning.",
#         "Ban illegal logging activities.",
#         "Raise awareness about conservation.",
#         "Reduce paper and wood consumption.",
#         "Implement carbon offset programs.",
#         "Support indigenous forest management."
#     ],
#     "Landslide": [
#         "Plant deep-rooted vegetation.",
#         "Construct retaining walls and terraces.",
#         "Improve drainage systems.",
#         "Implement early warning systems.",
#         "Enforce strict land-use policies.",
#         "Avoid deforestation in high-risk areas.",
#         "Use slope stabilization techniques.",
#         "Develop hazard maps for planning.",
#         "Monitor soil movement using sensors.",
#         "Train communities on emergency response."
#     ],
#     "Flood": [
#         "Improve urban drainage systems.",
#         "Construct flood barriers and levees.",
#         "Promote rainwater harvesting.",
#         "Implement sustainable land management.",
#         "Enforce floodplain zoning regulations.",
#         "Strengthen embankments along rivers.",
#         "Develop community flood response plans.",
#         "Use permeable surfaces in urban areas.",
#         "Implement reforestation in watersheds.",
#         "Deploy real-time flood monitoring systems."
#     ]
# }

# def process_image(image_path):
#     try:
#         img = load_img(image_path)  # Load original image
#         img_array = img_to_array(img) / 255.0  # Normalize
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#         predictions = model.predict(img_array)

#         predicted_mask = predictions[0]
#         predicted_class_index = np.argmax(predictions[1])
#         confidence_score = np.max(predictions[1]) * 100  # Convert to percentage

#         # Convert predicted mask to an image
#         predicted_mask = (predicted_mask.squeeze() * 255).astype(np.uint8)

#         # Convert grayscale to RGB if needed
#         if len(predicted_mask.shape) == 2:
#             predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2RGB)

#         # 🔥 Increase the size of both images (Original & Mask)
#         new_size = (512, 512)  # Adjust the size as needed

#         # Resize original image
#         original_img = cv2.imread(image_path)
#         original_resized = cv2.resize(original_img, new_size, interpolation=cv2.INTER_LINEAR)

#         # Save resized original image
#         resized_image_path = os.path.join(app.config["UPLOAD_FOLDER"], f"resized_{os.path.basename(image_path)}")
#         cv2.imwrite(resized_image_path, original_resized)

#         # Resize predicted mask
#         predicted_mask = cv2.resize(predicted_mask, new_size, interpolation=cv2.INTER_NEAREST)

#         # Save the mask
#         mask_filename = f"mask_{os.path.basename(image_path)}"
#         mask_path = os.path.join(app.config["MASK_FOLDER"], mask_filename)
#         cv2.imwrite(mask_path, predicted_mask)

#         predicted_disaster = disaster_classes[predicted_class_index]

#         return resized_image_path, mask_path, predicted_disaster, confidence_score
#     except Exception as e:
#         print(f"[ERROR] Image processing failed: {e}")
#         return None, None, None, None

# @app.route('/upload', methods=['POST'])
# def upload():
#     if 'files[]' not in request.files:
#         return jsonify({"error": "No files uploaded"}), 400

#     files = request.files.getlist('files[]')
#     results = []

#     for file in files:
#         if file.filename == '':
#             continue

#         file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#         file.save(file_path)

#         resized_image_path, mask_path, predicted_disaster, confidence_score = process_image(file_path)

#         if resized_image_path and mask_path and predicted_disaster:
#             results.append({
#                 "original_url": f"http://127.0.0.1:5000/uploads/{os.path.basename(resized_image_path)}",
#                 "mask_url": f"http://127.0.0.1:5000/masks/{os.path.basename(mask_path)}",
#                 "disaster_class": predicted_disaster,
#                 "accuracy": confidence_score
#             })
#         else:
#             return jsonify({"error": "Processing failed for some images"}), 500

#     return jsonify({"results": results})


# @app.route('/uploads/<filename>')
# def get_original(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# @app.route('/masks/<filename>')
# def get_mask(filename):
#     return send_from_directory(app.config["MASK_FOLDER"], filename)

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import cv2
from auth import auth_bp  # Your existing auth Blueprint

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)  # Allow all origins

app.register_blueprint(auth_bp)

# Define folders
UPLOAD_FOLDER = "uploads"
MASK_FOLDER = "masks"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MASK_FOLDER"] = MASK_FOLDER

MODEL_PATH = "disaster_multi_output_model.keras"
IMG_SIZE = (256, 256)

try:
    model = load_model(MODEL_PATH)
    print("[INFO] Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

disaster_classes = ["Deforestation", "Landslide", "Flood"]

def process_image(image_path):
    try:
        print(f"[DEBUG] Processing Image: {image_path}")
        if not os.path.exists(image_path):
            return None, None, None, None

        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)

        predicted_mask = predictions[0]
        predicted_class_index = np.argmax(predictions[1])
        confidence_score = np.max(predictions[1]) * 100

        predicted_mask = (predicted_mask.squeeze() * 255).astype(np.uint8)
        if len(predicted_mask.shape) == 2:
            predicted_mask = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2RGB)

        original_img = cv2.imread(image_path)
        if original_img is None:
            return None, None, None, None

        display_size = (512, 512)
        original_resized = cv2.resize(original_img, display_size)
        cv2.imwrite(image_path, original_resized)

        predicted_mask_resized = cv2.resize(predicted_mask, display_size)
        mask_filename = f"mask_{os.path.basename(image_path)}"
        mask_path = os.path.join(app.config["MASK_FOLDER"], mask_filename)
        cv2.imwrite(mask_path, predicted_mask_resized)

        predicted_disaster = disaster_classes[predicted_class_index]
        return image_path, mask_path, predicted_disaster, confidence_score
    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")
        return None, None, None, None

@app.route('/upload', methods=['POST'])
def upload():
    if 'files[]' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist('files[]')
    results = []

    for file in files:
        if file.filename == '':
            continue

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        resized_image_path, mask_path, predicted_disaster, confidence_score = process_image(file_path)

        if resized_image_path and mask_path and predicted_disaster:
            results.append({
                "original_url": request.host_url + f"uploads/{os.path.basename(resized_image_path)}",
                "mask_url": request.host_url + f"masks/{os.path.basename(mask_path)}",
                "disaster_class": predicted_disaster,
                "accuracy": confidence_score
            })
        else:
            return jsonify({"error": "Processing failed"}), 500

    return jsonify({"results": results})

@app.route('/uploads/<filename>')
def get_original(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/masks/<filename>')
def get_mask(filename):
    return send_from_directory(app.config["MASK_FOLDER"], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[INFO] Starting Flask on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
