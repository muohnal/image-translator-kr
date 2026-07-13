# Image English to Korean Translator

업로드한 이미지에서 영어 텍스트를 읽고 한국어로 번역하는 간단한 파이썬 앱입니다.

## 기능

- 이미지 업로드 (여러 장 지원, 진행률 표시)
- OCR로 영어 텍스트 추출 (작은 글씨 자동 확대 재인식)
- 한국어 번역 결과 표시 (배치 번역으로 빠른 처리, 동일 파일 캐싱)
- 번역 텍스트가 원문 위에 표시된 미리보기 이미지 제공 (다크모드 스크린샷 배경색 자동 매칭)
- 결과 표 및 CSV 다운로드

## 파일 구조

| 파일 | 역할 |
|---|---|
| `app.py` | Streamlit UI (업로드, 진행률, 결과 표시) |
| `ocr.py` | Tesseract OCR 전처리 및 라인 추출 |
| `translation.py` | Google 번역 (deep-translator) 배치 호출 |
| `rendering.py` | 번역 오버레이 미리보기 이미지 생성 |
| `export.py` | 결과 표/CSV 생성 (formula injection 방지 포함) |

## 실행 방법

1. Python 3.10+ 환경을 준비합니다.
2. 패키지를 설치합니다.

```bash
pip install -r requirements.txt
```

3. Tesseract OCR을 설치합니다.

- Windows 예시 경로: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- 설치 후 PATH에 등록하거나, PATH 등록이 어려우면 환경변수 `TESSERACT_CMD`에 실행 파일 경로를 지정합니다.
- 설치 확인: 터미널에서 `tesseract --version`을 실행해 정상 동작하는지 미리 확인하세요.

```powershell
# PATH에 없을 때 (PowerShell 예시)
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

4. 앱을 실행합니다.

```bash
streamlit run app.py
```

Windows에서는 `RunApp.bat`을 더블클릭해도 됩니다.

## Streamlit Cloud 배포

- `packages.txt`에 정의된 `tesseract-ocr`, `tesseract-ocr-kor`, `fonts-nanum`이 자동 설치되므로 별도 설정이 필요 없습니다.

## 참고

- OCR 정확도는 이미지 해상도와 글꼴 상태에 따라 달라집니다.
- 번역은 `deep-translator`(Google 번역)를 사용하므로 인터넷 연결이 필요하며, 추출된 텍스트가 외부 서비스로 전송됩니다. 민감한 정보가 담긴 이미지는 업로드하지 마세요.
- 업로드 파일은 15MB, 해상도는 약 4,000만 픽셀로 제한됩니다.
