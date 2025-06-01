import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def get_token():
    res = client.post("/token", data={"username": "admin", "password": "1234"})
    assert res.status_code == 200
    return res.json()["access_token"]

def test_login_success():
    token = get_token()
    assert token is not None

def test_login_fail():
    res = client.post("/token", data={"username": "wrong", "password": "wrong"})
    assert res.status_code == 400
    assert res.json()["detail"] == "Invalid credentials"

def test_predict_valid_image():
    token = get_token()
    file_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
    with open(file_path, "rb") as img:
        res = client.post(
            "/predict/",
            files={"file": ("test_image.jpg", img, "image/jpeg")},
            headers={"Authorization": f"Bearer {token}"}
        )
    # 🔍 디버깅용 출력
    print("STATUS:", res.status_code)
    print("RESPONSE:", res.json())
    
    assert res.status_code == 200
    assert "predicted_class" in res.json()

def test_predict_invalid_file():
    token = get_token()
    res = client.post(
        "/predict/",
        files={"file": ("fake.txt", b"not an image", "text/plain")},
        headers={"Authorization": f"Bearer {token}"}
    )
    print("INVALID RESPONSE:", res.status_code, res.json())
    assert res.status_code in [400, 500]
