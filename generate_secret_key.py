# generate_secret_key.py
import os
import base64

# 32바이트 길이의 무작위 바이트열 생성
# 이 길이는 일반적인 보안 권장 사항에 따릅니다.
random_bytes = os.urandom(32)

# URL-safe base64로 인코딩하여 문자열로 변환
# 웹 환경에서 사용하기 적합한 형태로 변환됩니다.
secret_key_bytes = base64.urlsafe_b64encode(random_bytes)
secret_key_string = secret_key_bytes.decode('utf-8')

print("생성된 SECRET_KEY:")
print(secret_key_string)