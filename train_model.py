# # import os
# # import numpy as np
# # import cv2
# # from glob import glob

# # # Base dataset directory
# # BASE_DIR = r"F:\disaster\dataset"

# # # Define paths for each disaster type and their respective image formats
# # DISASTER_TYPES = {
# #     "deforestationdata": "png",
# #     "landslidedata": "png",
# #     "flooddata": "jpg"  # Flood data has JPG images
# # }

# # # Function to load dataset
# # def load_dataset():
# #     X, Y = [], []  # X -> Images, Y -> Masks

# #     for disaster, img_ext in DISASTER_TYPES.items():
# #         image_dir = os.path.join(BASE_DIR, disaster, "images")
# #         mask_dir = os.path.join(BASE_DIR, disaster, "masks")

# #         # Load images and masks
# #         image_paths = glob(os.path.join(image_dir, f"*.{img_ext}"))  # Image extension varies
# #         mask_paths = glob(os.path.join(mask_dir, "*.png"))  # Masks are always PNG

# #         # Sort to maintain order
# #         image_paths.sort()
# #         mask_paths.sort()

# #         for img_path, mask_path in zip(image_paths, mask_paths):
# #             img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Read image
# #             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask

# #             if img is None or mask is None:
# #                 print(f"Error loading {img_path} or {mask_path}")
# #                 continue  # Skip if file not found

# #             img = cv2.resize(img, (256, 256))  # Resize for consistency
# #             mask = cv2.resize(mask, (256, 256))

# #             X.append(img)
# #             Y.append(mask)

# #     # Convert to NumPy arrays
# #     X = np.array(X, dtype=np.float32) / 255.0  # Normalize images
# #     Y = np.array(Y, dtype=np.float32)  # Keep masks as is

# #     # Expand mask dimensions if necessary
# #     if len(Y.shape) == 3:
# #         Y = np.expand_dims(Y, axis=-1)

# #     print(f"Dataset Loaded: Images {X.shape}, Masks {Y.shape}")
# #     return X, Y
# # X, Y = load_dataset()
# # print("Images Shape:", X.shape)
# # print("Masks Shape:", Y.shape)


# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import load_img, img_to_array

# def DATASET_PATH():
#     return {
#         "deforestation": {
#             "images": "F:/disaster/dataset/deforestationdata/images",
#             "masks": "F:/disaster/dataset/deforestationdata/masks"
#         },
#         "landslide": {
#             "images": "F:/disaster/dataset/landslidedata/images",
#             "masks": "F:/disaster/dataset/landslidedata/masks"
#         },
#         "flood": {
#             "images": "F:/disaster/dataset/flooddata/images",
#             "masks": "F:/disaster/dataset/flooddata/masks"
#         }
#     }

# def load_dataset(img_size=(256, 256)):
#     dataset_paths = DATASET_PATH()
#     X, Y = [], []
    
#     for disaster, paths in dataset_paths.items():
#         img_dir, mask_dir = paths['images'], paths['masks']
#         image_filenames = sorted(os.listdir(img_dir))
#         mask_filenames = sorted(os.listdir(mask_dir))
        
#         for img_filename, mask_filename in zip(image_filenames, mask_filenames):
#             img_path = os.path.join(img_dir, img_filename)
#             mask_path = os.path.join(mask_dir, mask_filename)
            
#             try:
#                 img = load_img(img_path, target_size=img_size)
#                 mask = load_img(mask_path, target_size=img_size, color_mode='grayscale')
#                 X.append(img_to_array(img) / 255.0)
#                 Y.append(img_to_array(mask) / 255.0)
#             except Exception as e:
#                 print(f"Error loading {img_path} or {mask_path}: {e}")
    
#     X = np.array(X, dtype=np.float32)
#     Y = np.array(Y, dtype=np.float32).reshape(-1, img_size[0], img_size[1], 1)
#     print(f"Dataset Loaded: Images {X.shape}, Masks {Y.shape}")
#     return X, Y

# def build_model(input_shape=(256, 256, 3)):
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
#         layers.UpSampling2D((2, 2)),
#         layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
#         layers.UpSampling2D((2, 2)),
#         layers.Conv2D(1, (1, 1), activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# if __name__ == "__main__":
#     X, Y = load_dataset()
#     model = build_model()
#     model.fit(X, Y, epochs=10, batch_size=8, validation_split=0.2)
#     model.save("disaster_segmentation_model.h5")
#     print("Model training complete and saved!")




# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from sklearn.preprocessing import LabelEncoder

# # Disaster categories
# DISASTER_CATEGORIES = ["deforestation", "landslide", "flood"]

# def DATASET_PATH():
#     return {
#         "deforestation": {
#             "images": "F:/disaster/dataset/deforestationdata/images",
#             "masks": "F:/disaster/dataset/deforestationdata/masks"
#         },
#         "landslide": {
#             "images": "F:/disaster/dataset/landslidedata/images",
#             "masks": "F:/disaster/dataset/landslidedata/masks"
#         },
#         "flood": {
#             "images": "F:/disaster/dataset/flooddata/images",
#             "masks": "F:/disaster/dataset/flooddata/masks"
#         }
#     }

# def load_dataset(img_size=(256, 256)):
#     dataset_paths = DATASET_PATH()
#     X, Y, labels = [], [], []
    
#     for disaster, paths in dataset_paths.items():
#         img_dir, mask_dir = paths['images'], paths['masks']
#         image_filenames = sorted(os.listdir(img_dir))
#         mask_filenames = sorted(os.listdir(mask_dir))
        
#         for img_filename, mask_filename in zip(image_filenames, mask_filenames):
#             img_path = os.path.join(img_dir, img_filename)
#             mask_path = os.path.join(mask_dir, mask_filename)
            
#             try:
#                 img = load_img(img_path, target_size=img_size)
#                 mask = load_img(mask_path, target_size=img_size, color_mode='grayscale')
                
#                 X.append(img_to_array(img) / 255.0)  # Normalize images
#                 Y.append(img_to_array(mask) / 255.0)  # Normalize masks
                
#                 # Store label as category name
#                 labels.append(disaster)
#             except Exception as e:
#                 print(f"Error loading {img_path} or {mask_path}: {e}")

#     # Convert lists to numpy arrays
#     X = np.array(X, dtype=np.float32)
#     Y = np.array(Y, dtype=np.float32).reshape(-1, img_size[0], img_size[1], 1)

#     # Encode labels
#     label_encoder = LabelEncoder()
#     labels = label_encoder.fit_transform(labels)  # Convert to numeric labels
#     labels = tf.keras.utils.to_categorical(labels, num_classes=len(DISASTER_CATEGORIES))

#     print(f"Dataset Loaded: Images {X.shape}, Masks {Y.shape}, Labels {labels.shape}")
#     return X, Y, labels

# def build_model(input_shape=(256, 256, 3), num_classes=3):
#     inputs = layers.Input(shape=input_shape)

#     # Shared Convolutional Layers
#     x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
#     x = layers.MaxPooling2D((2, 2))(x)
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = layers.MaxPooling2D((2, 2))(x)

#     # Segmentation Output
#     x_seg = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
#     x_seg = layers.UpSampling2D((2, 2))(x_seg)
#     x_seg = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x_seg)
#     x_seg = layers.UpSampling2D((2, 2))(x_seg)
#     seg_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name="segmentation_output")(x_seg)

#     # Classification Output
#     x_cls = layers.GlobalAveragePooling2D()(x)  # Convert feature map to a vector
#     x_cls = layers.Dense(64, activation='relu')(x_cls)
#     class_output = layers.Dense(num_classes, activation='softmax', name="classification_output")(x_cls)

#     # Define model with two outputs
#     model = models.Model(inputs=inputs, outputs=[seg_output, class_output])

#     # Compile model
#     model.compile(optimizer='adam',
#                   loss={'segmentation_output': 'binary_crossentropy', 'classification_output': 'categorical_crossentropy'},
#                   metrics={'segmentation_output': 'accuracy', 'classification_output': 'accuracy'})

#     return model

# if __name__ == "__main__":
#     X, Y, labels = load_dataset()
#     model = build_model()

#     model.fit(X, {"segmentation_output": Y, "classification_output": labels}, 
#               epochs=10, batch_size=8, validation_split=0.2)

#     model.save("disaster_model.h5")
#     print("Model training complete and saved!")


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Define the multi-output model
def build_multi_output_model(input_shape=(256, 256, 3), num_classes=3):
    inputs = Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Bottleneck
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Decoder (Segmentation Output)
    y = layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    y = layers.UpSampling2D((2, 2))(y)
    y = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(y)
    y = layers.UpSampling2D((2, 2))(y)
    segmentation_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name="segmentation_output")(y)

    # Classification Output
    c = layers.GlobalAveragePooling2D()(x)
    c = layers.Dense(64, activation='relu')(c)
    classification_output = layers.Dense(num_classes, activation='softmax', name="classification_output")(c)

    # Create model
    model = models.Model(inputs=inputs, outputs=[segmentation_output, classification_output])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss={
            "segmentation_output": "binary_crossentropy",
            "classification_output": "categorical_crossentropy"
        },
        metrics={
            "segmentation_output": ["accuracy"],
            "classification_output": ["accuracy"]
        }
    )
    
    return model

# Dataset paths
DATASET_PATHS = {
    "deforestation": {
        "images": "F:/disaster/dataset/deforestationdata/images",
        "masks": "F:/disaster/dataset/deforestationdata/masks"
    },
    "landslide": {
        "images": "F:/disaster/dataset/landslidedata/images",
        "masks": "F:/disaster/dataset/landslidedata/masks"
    },
    "flood": {
        "images": "F:/disaster/dataset/flooddata/images",
        "masks": "F:/disaster/dataset/flooddata/masks"
    }
}

# Load dataset
def load_dataset(img_size=(256, 256), num_classes=3):
    X, Y_segmentation, Y_classification = [], [], []
    disaster_labels = {"deforestation": 0, "landslide": 1, "flood": 2}

    for disaster, paths in DATASET_PATHS.items():
        img_dir, mask_dir = paths['images'], paths['masks']
        image_filenames = sorted(os.listdir(img_dir))
        mask_filenames = sorted(os.listdir(mask_dir))
        
        for img_filename, mask_filename in zip(image_filenames, mask_filenames):
            img_path = os.path.join(img_dir, img_filename)
            mask_path = os.path.join(mask_dir, mask_filename)
            
            try:
                img = load_img(img_path, target_size=img_size)
                mask = load_img(mask_path, target_size=img_size, color_mode='grayscale')
                X.append(img_to_array(img) / 255.0)
                Y_segmentation.append(img_to_array(mask) / 255.0)
                Y_classification.append(disaster_labels[disaster])
            except Exception as e:
                print(f"Error loading {img_path} or {mask_path}: {e}")

    X = np.array(X, dtype=np.float32)
    Y_segmentation = np.array(Y_segmentation, dtype=np.float32).reshape(-1, img_size[0], img_size[1], 1)
    Y_classification = to_categorical(Y_classification, num_classes=num_classes)

    print(f"Dataset Loaded: Images {X.shape}, Masks {Y_segmentation.shape}, Classes {Y_classification.shape}")
    return X, Y_segmentation, Y_classification

# Train the model
if __name__ == "__main__":
    X, Y_segmentation, Y_classification = load_dataset()
    model = build_multi_output_model()

    model.fit(
        X, 
        {"segmentation_output": Y_segmentation, "classification_output": Y_classification},
        epochs=10,
        batch_size=8,
        validation_split=0.2
    )

    model.save("disaster_multi_output_model.keras")
    print("✅ Model training complete and saved!")

# Prediction function
def predict_disaster(image_path, model):
    img_size = (256, 256)
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    segmentation_output, classification_output = model.predict(img_array)

    predicted_class = np.argmax(classification_output)
    class_labels = ["Deforestation", "Landslide", "Flood"]
    
    print(f"🛑 Predicted Disaster Class: {class_labels[predicted_class]}")
    
    return segmentation_output[0], class_labels[predicted_class]

# Visualization function
def visualize_prediction(image_path, model):
    segmentation_mask, disaster_class = predict_disaster(image_path, model)
    
    original_img = load_img(image_path, target_size=(256, 256))
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation_mask.squeeze(), cmap="gray")
    plt.title(f"Predicted Segmentation ({disaster_class})")
    
    plt.show()

# Example usage
if __name__ == "__main__":
    model = tf.keras.models.load_model("disaster_multi_output_model.keras")
    visualize_prediction("7_Balak.png", model)
