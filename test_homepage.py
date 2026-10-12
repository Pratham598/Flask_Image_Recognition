
"""Tests for the Flask application homepage."""


def test_homepage_accessible(client):
    """Verify the homepage loads and displays the upload form."""
    response = client.get("/")

    assert response.status_code == 200
    assert b"Hand Sign Digit Language Detection" in response.data
    assert b'name="file"' in response.data
