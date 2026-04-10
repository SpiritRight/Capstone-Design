from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def load_env_file(env_path: Path) -> None:
    # .env 파일이 있으면 현재 프로세스 환경변수로 로드한다.
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


def _env_str(name: str, default: str = "") -> str:
    # 문자열 설정은 비어 있으면 기본값으로 처리한다.
    value = os.getenv(name)
    return default if value in (None, "") else value


def _env_int(name: str, default: int) -> int:
    # 숫자 설정은 여기서 한 번만 변환해 두고 이후에는 그대로 사용한다.
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    # float 설정도 별도 helper로 읽어 상단 설정 로직을 단순하게 유지한다.
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return float(value)


# improved_baseline/.env를 기본 설정 파일로 사용한다.
load_env_file(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    # 파이프라인 전체에서 공유하는 설정 값을 한 객체에 모아둔다.
    base_url: str
    config_url: str
    upload_url: str
    refresh_url: str
    login_url: str
    capture_root: str
    auth_provider: str
    login_id: str
    login_password: str
    access_token: str
    refresh_token: str
    token_refresh_margin_seconds: int
    request_timeout_seconds: int
    chrome_login_wait_seconds: int
    movement_threshold: float
    stationary_frames_threshold: int
    match_distance_threshold: float
    max_missing_frames: int
    area_growth_ratio: float
    max_stored_frames: int
    fps: int
    pre_frames: int
    post_frames: int
    stale_timeout_seconds: int
    poll_retry_seconds: int
    sse_connect_timeout_seconds: int
    sse_read_timeout_seconds: int
    frame_width: int
    frame_height: int
    save_width: int
    save_height: int
    rtsp_port: int
    detection_confidence: float
    yolo_model_path: str
    log_level: str


def load_settings() -> Settings:
    # 환경변수를 실제 실행 코드에서 직접 읽지 않도록 여기서 한 번에 정리한다.
    base_url = _env_str("CG_BASE_URL", "http://3.36.174.53:8080")
    return Settings(
        base_url=base_url,
        config_url=_env_str("CG_CONFIG_URL", f"{base_url}/cleanguard/cctv/sse"),
        upload_url=_env_str("CG_UPLOAD_URL", f"{base_url}/cleanguard/image/"),
        refresh_url=_env_str("CG_REFRESH_URL", f"{base_url}/auth/refresh"),
        login_url=_env_str("CG_LOGIN_URL", f"{base_url}/oauth2/authorization/kakao"),
        capture_root=_env_str("CG_CAPTURE_ROOT", "captures"),
        auth_provider=_env_str("CG_AUTH_PROVIDER", "refresh").lower(),
        login_id=_env_str("LOGIN_ID"),
        login_password=_env_str("LOGIN_PASSWORD"),
        access_token=_env_str("CG_ACCESS_TOKEN"),
        refresh_token=_env_str("CG_REFRESH_TOKEN"),
        token_refresh_margin_seconds=_env_int("CG_TOKEN_REFRESH_MARGIN_SECONDS", 60),
        request_timeout_seconds=_env_int("CG_REQUEST_TIMEOUT_SECONDS", 10),
        chrome_login_wait_seconds=_env_int("CG_CHROME_LOGIN_WAIT_SECONDS", 5),
        movement_threshold=_env_float("CG_TRACK_MOVEMENT_THRESHOLD", 10.0),
        stationary_frames_threshold=_env_int("CG_TRACK_STATIONARY_FRAMES", 5),
        match_distance_threshold=_env_float("CG_TRACK_MATCH_DISTANCE", 30.0),
        max_missing_frames=_env_int("CG_TRACK_MAX_MISSING_FRAMES", 15),
        area_growth_ratio=_env_float("CG_TRACK_AREA_GROWTH_RATIO", 1.1),
        max_stored_frames=_env_int("CG_MAX_STORED_FRAMES", 1000),
        fps=_env_int("CG_RECORD_FPS", 5),
        pre_frames=_env_int("CG_RECORD_PRE_FRAMES", 35),
        post_frames=_env_int("CG_RECORD_POST_FRAMES", 35),
        stale_timeout_seconds=_env_int("CG_POLLER_STALE_TIMEOUT", 60),
        poll_retry_seconds=_env_int("CG_POLLER_RETRY_SECONDS", 5),
        sse_connect_timeout_seconds=_env_int("CG_POLLER_CONNECT_TIMEOUT", 5),
        sse_read_timeout_seconds=_env_int("CG_POLLER_READ_TIMEOUT", 5),
        frame_width=_env_int("CG_FRAME_WIDTH", 2304),
        frame_height=_env_int("CG_FRAME_HEIGHT", 1296),
        save_width=_env_int("CG_SAVE_WIDTH", 640),
        save_height=_env_int("CG_SAVE_HEIGHT", 480),
        rtsp_port=_env_int("CG_RTSP_PORT", 554),
        detection_confidence=_env_float("CG_DETECTOR_CONFIDENCE", 0.5),
        yolo_model_path=_env_str("CG_MODEL_PATH", "yolo11m_train.engine"),
        log_level=_env_str("CG_LOG_LEVEL", "INFO").upper(),
    )
