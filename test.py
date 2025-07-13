import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('disaster_segmentation_model.h5')

# Preprocessing function (Move it above visualize_prediction)
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(256, 256))  # Resize image
    img = img_to_array(img) / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to visualize the prediction
def visualize_prediction(image_path, mask_path):
    img = load_img(image_path, target_size=(256, 256))
    mask = load_img(mask_path, target_size=(256, 256), color_mode="grayscale")

    # Call preprocess_image correctly
    pred_mask = model.predict(preprocess_image(image_path))[0]

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Ground Truth Mask")
    ax[2].imshow(pred_mask.squeeze(), cmap="gray")
    ax[2].set_title("Predicted Mask")
    plt.show()

# Test the function
visualize_prediction('img (2).png', 'img (1).png')

# Print model summary
print(model.summary())

# Test with a sample image
test_image = preprocess_image('img (2).png')
prediction = model.predict(test_image)
print("Prediction Output:", prediction)
