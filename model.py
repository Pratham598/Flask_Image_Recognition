
"""Image preprocessing and prediction functions for hand-sign recognition."""

from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image


# Load the trained digit recognition model.
model = load_model("digit_model.h5")


def preprocess_img(img_path):
    """Resize and normalize an uploaded image for model prediction."""
    op_img = Image.open(img_path)
    img_resize = op_img.resize((224, 224))
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, 224, 224, 3)
    return img_reshape


def predict_result(predict):
    """Return the digit with the highest prediction score."""
    pred = model.predict(predict)
    return np.argmax(pred[0], axis=-1)
