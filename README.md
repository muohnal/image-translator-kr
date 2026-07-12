# Image English to Korean Translator

업로드한 이미지에서 영어 텍스트를 읽고 한국어로 번역하는 간단한 파이썬 앱입니다.

## 기능

- 이미지 업로드
- OCR로 영어 텍스트 추출
- 한국어 번역 결과 표시
- 번역 텍스트가 표시된 미리보기 이미지 제공
- CSV 다운로드

## 실행 방법

1. Python 3.10+ 환경을 준비합니다.
2. 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

3. Tesseract OCR을 설치합니다.

- Windows 예시 경로: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- 설치 후 앱 사이드바에 경로를 넣거나 PATH에 등록합니다.
- 설치 확인: 터미널에서 `tesseract --version`을 실행해 정상 동작하는지 미리 확인하세요.

4. 앱을 실행합니다.

```bash
streamlit run app.py
```

## 참고

- OCR 정확도는 이미지 해상도와 글꼴 상태에 따라 달라집니다.
- 번역은 `deep-translator`(Google 번역)를 사용하므로 인터넷 연결이 필요하며, 추출된 텍스트가 외부 서비스로 전송됩니다. 민감한 정보가 담긴 이미지는 업로드하지 마세요.
- 업로드 파일은 15MB, 해상도는 약 4,000만 픽셀로 제한됩니다.
