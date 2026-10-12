
"""Acceptance tests for successful image uploads."""

from io import BytesIO

from PIL import Image


def create_test_image(size=(224, 224)):
    """Create a valid RGB image in memory."""
    image = Image.new("RGB", size, color="white")
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes


def test_acceptance_successful_upload(client, monkeypatch):
    """Verify a valid upload displays the predicted digit."""
    monkeypatch.setattr("app.predict_result", lambda image: 3)

    response = client.post(
        "/prediction",
        data={"file": (create_test_image(), "test_image.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." not in response.data
    assert b">3</h2>" in response.data


def test_acceptance_valid_large_image(client, monkeypatch):
    """Verify a large valid image is accepted and processed."""
    monkeypatch.setattr("app.predict_result", lambda image: 5)

    response = client.post(
        "/prediction",
        data={"file": (create_test_image((1024, 1024)), "large_image.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." not in response.data
    assert b">5</h2>" in response.data


def test_acceptance_valid_image_size_upload(client, monkeypatch):
    """Verify a small valid image is accepted and processed."""
    monkeypatch.setattr("app.predict_result", lambda image: 7)

    response = client.post(
        "/prediction",
        data={"file": (create_test_image((64, 64)), "small_image.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." not in response.data
    assert b">7</h2>" in response.data
