# Capstone-Design

YOLO 기반 CCTV 무단투기 탐지 캡스톤 프로젝트입니다.

이 저장소는 원본 베이스라인 코드와, 캡스톤 종료 후 유지보수성을 고려해 정리한 개선 버전을 함께 포함합니다.

## 프로젝트 개요

이 프로젝트의 목표는 CCTV 영상에서 쓰레기봉투 개수를 정확히 세는 것이 아니라, 특정 구역에서 무단투기 이벤트가 발생했는지를 판단하는 것입니다.

전체 파이프라인은 아래 순서로 동작합니다.

1. 서버에서 SSE로 활성 CCTV 목록을 수신
2. CCTV별 RTSP 스트림을 별도 프로세스로 수신
3. YOLO로 프레임 단위 객체 탐지
4. bbox 중심 거리와 정지 시간 기준으로 후보 추적
5. 일정 프레임 이상 방치된 경우 이벤트로 판단
6. 이벤트 전후 프레임을 mp4로 저장 후 서버 업로드

## 저장소 구성

- [`baseline_original.py`](./baseline_original.py)
  - 캡스톤 당시 사용한 원본 베이스라인 코드
  - 인증, SSE poller, worker, 탐지, 업로드가 하나의 파일에 모여 있는 구조
- [`improved_baseline`](./improved_baseline)
  - 원본 구조는 유지하면서 설정과 인증을 분리한 개선 버전
  - `config.py`, `auth.py`, `main.py`로 최소한만 분리

## 개선 버전에서 정리한 부분

- 하드코딩된 URL, 경로, 탐지 파라미터를 `.env`와 설정 객체로 분리
- 메인 파이프라인에서 인증 로직을 분리
- `refresh token`, `env token`, `selenium` 인증 방식을 선택 가능하도록 구성
- `print` 중심 출력을 `logging` 기반으로 정리
- 원본의 bbox 기반 후보 추적 로직은 유지

즉, 개선 버전은 알고리즘을 새로 바꾼 것이 아니라 원본 파이프라인을 조금 더 설명 가능하고 유지보수 가능한 형태로 정리한 버전입니다.

## 왜 YOLO tracking 대신 직접 추적했는가

초기 실험에서 YOLO tracking은 작은 객체나 CCTV 환경에서 ID switching이 자주 발생했습니다.

이 프로젝트의 핵심 목표는 객체별 ID를 장기간 정확히 유지하는 것이 아니라, 같은 위치에 일정 시간 이상 쓰레기가 방치되었는지를 판단하는 것이었습니다. 그래서 범용 tracking 대신 아래 기준으로 직접 후보를 관리했습니다.

- bbox 중심 거리
- 일정 프레임 이상 정지 여부
- 잠시 detection이 끊겨도 유지하는 grace frame
- 크기 급변 시 stationary count 초기화

이 방식은 여러 봉투가 가까이 있을 때 개별 구분은 약할 수 있지만, 본 프로젝트의 목표인 무단투기 이벤트 검출에는 충분히 타당하다고 판단했습니다.

## 개선 버전 실행 방법

### 1. 환경 준비

필요한 주요 도구는 아래와 같습니다.

- Python 3
- `ffmpeg`
- OpenCV
- `ultralytics`
- `sseclient`
- `requests`
- `ffmpeg-python`
- Selenium 사용 시 Chrome 및 webdriver 관련 패키지

### 2. 환경변수 파일 준비

```bash
cd /mnt/d/Capstone/improved_baseline
cp .env.example .env
```

기본적으로는 `refresh token` 기반 인증을 사용하도록 작성했습니다.

최소한 아래 값은 실제 환경에 맞게 수정해야 합니다.

```env
CG_AUTH_PROVIDER=refresh
CG_BASE_URL=""
CG_REFRESH_TOKEN=your_refresh_token
CG_MODEL_PATH=yolo11m_train.engine
```

### 3. 실행

```bash
cd /mnt/d/Capstone
python3 improved_baseline/main.py
```

## 주요 코드 설명

- [`improved_baseline/config.py`](./improved_baseline/config.py)
  - `.env`를 읽고 실행 설정을 `Settings` 객체로 정리
- [`improved_baseline/auth.py`](./improved_baseline/auth.py)
  - 인증 방식 분리
  - `AuthorizedSession`을 통해 토큰 자동 부착 및 401 재시도 처리
- [`improved_baseline/main.py`](./improved_baseline/main.py)
  - 메인 프로세스, SSE poller, CCTV별 worker, YOLO 추론, 이벤트 저장/업로드 담당

## 한계

- 실제 운영 환경 전체를 이 저장소만으로 재현할 수는 없습니다.
- 백엔드 인증 정책과 CCTV 스트림 환경이 준비되어 있어야 동작합니다.
- bbox 거리 기반의 단순 추적 방식이기 때문에 겹치는 객체를 정밀하게 구분하는 데는 한계가 있습니다.
- threshold 값은 카메라 화각과 환경에 따라 재조정이 필요합니다.


