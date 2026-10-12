
"""Acceptance tests for invalid inputs and edge cases."""

from io import BytesIO

from PIL import Image


def create_image(size=(224, 224), image_format="PNG"):
    """Create a valid image in memory for testing."""
    image = Image.new("RGB", size, color="white")
    image_bytes = BytesIO()
    image.save(image_bytes, format=image_format)
    image_bytes.seek(0)
    return image_bytes


def test_edge_case_large_image_upload(client, monkeypatch):
    """Verify a large valid image can be processed."""
    monkeypatch.setattr("app.predict_result", lambda image: 2)

    response = client.post(
        "/prediction",
        data={"file": (create_image((1024, 1024)), "large.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b">2</h2>" in response.data
    assert b"File cannot be processed." not in response.data


def test_edge_case_invalid_metadata(client):
    """Verify invalid image data produces an error message."""
    image_data = BytesIO(b"not_a_real_image")

    response = client.post(
        "/prediction",
        data={"file": (image_data, "invalid.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data


def test_edge_case_non_standard_image_extensions(client, monkeypatch):
    """Verify that a valid WebP image is accepted."""
    monkeypatch.setattr("app.predict_result", lambda image: 4)

    response = client.post(
        "/prediction",
        data={"file": (create_image(image_format="WEBP"), "image.webp")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b">4</h2>" in response.data


def test_edge_case_sequential_image_uploads(client, monkeypatch):
    """Verify two consecutive image uploads succeed."""
    monkeypatch.setattr("app.predict_result", lambda image: 6)

    for filename in ("first.png", "second.png"):
        response = client.post(
            "/prediction",
            data={"file": (create_image(), filename)},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        assert b">6</h2>" in response.data


def test_edge_case_unexpected_headers(client, monkeypatch):
    """Verify extra HTTP headers do not prevent valid uploads."""
    monkeypatch.setattr("app.predict_result", lambda image: 8)

    response = client.post(
        "/prediction",
        data={"file": (create_image(), "headers.png")},
        content_type="multipart/form-data",
        headers={"X-Unexpected-Header": "value"},
    )

    assert response.status_code == 200
    assert b">8</h2>" in response.data


def test_edge_case_empty_image_file(client):
    """Verify an empty uploaded image produces an error."""
    response = client.post(
        "/prediction",
        data={"file": (BytesIO(b""), "empty.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert b"File cannot be processed." in response.data
