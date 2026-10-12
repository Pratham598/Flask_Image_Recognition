
"""Unit tests for image preprocessing and digit prediction."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

import model as model_module


def make_image(size=(300, 200), color=(255, 128, 0)):
    """Create a valid RGB test image in memory."""
    image = Image.new("RGB", size, color=color)
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes


def test_preprocess_image_shape():
    """Verify preprocessing produces the expected model input shape."""
    processed = model_module.preprocess_img(make_image())

    assert processed.shape == (1, 224, 224, 3)


def test_preprocess_image_normalization():
    """Verify pixel values are normalized between zero and one."""
    processed = model_module.preprocess_img(make_image())

    assert np.min(processed) >= 0.0
    assert np.max(processed) <= 1.0
    assert np.isclose(processed[0, 0, 0, 0], 1.0)


def test_invalid_image_path():
    """Verify a nonexistent image path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        model_module.preprocess_img("invalid/path/to/image.jpeg")


def test_predict_result_selects_highest_score(monkeypatch):
    """Verify prediction returns the class with the highest score."""

    class FakeModel:
        """Provide predictable classification scores."""

        def predict(self, image):
            """Return scores where class four has the highest value."""
            return np.array(
                [[0.1, 0.05, 0.02, 0.1, 0.6, 0.03, 0.02, 0.03, 0.03, 0.02]]
            )

    monkeypatch.setattr(model_module, "model", FakeModel())

    image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    prediction = model_module.predict_result(image)

    assert isinstance(prediction, (int, np.integer))
    assert prediction == 4


def test_preprocess_grayscale_image():
    """Verify how preprocessing handles a grayscale image."""
    grayscale = Image.new("L", (224, 224), color=128)
    image_bytes = BytesIO()
    grayscale.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    with pytest.raises(ValueError):
        model_module.preprocess_img(image_bytes)
