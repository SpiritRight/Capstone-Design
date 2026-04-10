from __future__ import annotations

import base64
import json
import threading
import time
from urllib.parse import parse_qs, urlparse

import requests

try:
    from .config import Settings
except ImportError:
    from config import Settings


class TokenProvider:
    # 인증 방식이 달라도 아래 두 메서드만 맞추면 같은 방식으로 사용할 수 있다.
    def get_token(self, force_refresh: bool = False) -> str:
        raise NotImplementedError

    def invalidate(self) -> None:
        raise NotImplementedError


def _strip_bearer_prefix(token: str) -> str:
    # 입력값이 "Bearer xxx" 형태여도 내부에서는 토큰 원문만 다룬다.
    token = token.strip()
    if token.lower().startswith("bearer "):
        return token[7:].strip()
    return token


def _decode_jwt_exp(token: str) -> float | None:
    # access token 안의 만료 시각(exp)을 읽어 자동 갱신 시점을 계산한다.
    try:
        # JWT는 header.payload.signature 구조라서 가운데 payload만 디코딩하면 된다.
        header, payload, signature = token.split(".")
        del header, signature
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8"))
        data = json.loads(decoded.decode("utf-8"))
        exp = data.get("exp")
        return None if exp is None else float(exp)
    except Exception:
        return None


class EnvTokenProvider(TokenProvider):
    # 이미 발급된 access token을 그대로 사용하는 가장 단순한 방식이다.
    def __init__(self, access_token: str) -> None:
        token = _strip_bearer_prefix(access_token)
        if not token:
            raise RuntimeError("CG_ACCESS_TOKEN is required when CG_AUTH_PROVIDER=env")
        self._token = token

    def get_token(self, force_refresh: bool = False) -> str:
        return self._token

    def invalidate(self) -> None:
        return None


class RefreshTokenProvider(TokenProvider):
    # refresh token으로 access token을 자동 재발급받는 기본 인증 방식이다.
    def __init__(self, settings: Settings) -> None:
        refresh_token = _strip_bearer_prefix(settings.refresh_token)
        if not refresh_token:
            raise RuntimeError("CG_REFRESH_TOKEN is required when CG_AUTH_PROVIDER=refresh")

        self._refresh_url = settings.refresh_url
        self._request_timeout_seconds = settings.request_timeout_seconds
        self._refresh_margin_seconds = settings.token_refresh_margin_seconds
        self._refresh_token = refresh_token
        self._access_token = ""
        self._access_token_exp = 0.0
        self._lock = threading.Lock()

        if settings.access_token:
            self._store_access_token(settings.access_token)

    def _store_access_token(self, token: str) -> None:
        # 새 access token을 저장하면서 만료 시각도 같이 캐싱한다.
        normalized = _strip_bearer_prefix(token)
        exp = _decode_jwt_exp(normalized)
        self._access_token = normalized
        self._access_token_exp = exp or 0.0

    def _needs_refresh(self) -> bool:
        # 캐시된 토큰이 없거나, 곧 만료되면 재발급이 필요하다.
        if not self._access_token:
            return True
        if not self._access_token_exp:
            return False
        return time.time() >= self._access_token_exp - self._refresh_margin_seconds

    def _refresh(self) -> str:
        # 백엔드의 /auth/refresh 엔드포인트를 호출해 새 access token을 받는다.
        response = requests.post(
            self._refresh_url,
            json={},
            headers={"Refresh-Token": f"Bearer {self._refresh_token}"},
            timeout=self._request_timeout_seconds,
        )
        response.raise_for_status()

        payload = response.json()
        access_token = payload.get("access_token") or payload.get("accessToken")
        if not access_token:
            raise RuntimeError("Refresh endpoint did not return an access token")

        rotated_refresh = (
            payload.get("refresh_token")
            or payload.get("refreshToken")
            or response.headers.get("Refresh-Token")
        )
        if rotated_refresh:
            # 서버가 refresh token도 같이 교체해 주는 경우를 대비해 최신 값으로 갱신한다.
            self._refresh_token = _strip_bearer_prefix(rotated_refresh)

        self._store_access_token(access_token)
        return self._access_token

    def get_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            if force_refresh or self._needs_refresh():
                return self._refresh()
            return self._access_token

    def invalidate(self) -> None:
        with self._lock:
            self._access_token = ""
            self._access_token_exp = 0.0


class SeleniumTokenProvider(TokenProvider):
    # 초기 캡스톤 방식과의 호환을 위해 남겨둔 fallback 인증 방식이다.
    def __init__(self, settings: Settings) -> None:
        if not settings.login_id or not settings.login_password:
            raise RuntimeError("LOGIN_ID and LOGIN_PASSWORD are required when CG_AUTH_PROVIDER=selenium")

        self._login_url = settings.login_url
        self._login_id = settings.login_id
        self._login_password = settings.login_password
        self._wait_seconds = settings.chrome_login_wait_seconds
        self._token = ""
        self._lock = threading.Lock()

    def _fetch(self) -> str:
        # 브라우저 자동화를 통해 로그인 후 redirect URL에서 access token을 추출한다.
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from webdriver_manager.chrome import ChromeDriverManager

        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=service, options=options)

        try:
            driver.get(self._login_url)
            driver.find_element(By.NAME, "loginId").send_keys(self._login_id)
            driver.find_element(By.NAME, "password").send_keys(self._login_password, Keys.RETURN)
            time.sleep(self._wait_seconds)
            qs = parse_qs(urlparse(driver.current_url).query)
            token = qs.get("access_token", [None])[0]
            if not token:
                raise RuntimeError("No access token found after Selenium login")
            return _strip_bearer_prefix(token)
        finally:
            driver.quit()

    def get_token(self, force_refresh: bool = False) -> str:
        with self._lock:
            if force_refresh or not self._token:
                self._token = self._fetch()
            return self._token

    def invalidate(self) -> None:
        with self._lock:
            self._token = ""


class AuthorizedSession:
    # 모든 HTTP 요청에 Authorization 헤더를 붙이고, 401이면 한 번 더 재시도한다.
    def __init__(self, token_provider: TokenProvider) -> None:
        self._token_provider = token_provider
        self._session = requests.Session()

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        user_headers = dict(kwargs.pop("headers", {}) or {})
        for attempt in range(2):
            headers = dict(user_headers)
            headers["Authorization"] = f"Bearer {self._token_provider.get_token(force_refresh=attempt > 0)}"
            response = self._session.request(method, url, headers=headers, **kwargs)
            if response.status_code != 401:
                return response
            # 첫 요청이 401이면 access token이 만료된 것으로 보고 한 번만 강제 재발급 후 재시도한다.
            response.close()
            self._token_provider.invalidate()
        return response

    def get(self, url: str, **kwargs) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self.request("POST", url, **kwargs)


def build_token_provider(settings: Settings) -> TokenProvider:
    # 환경변수에 지정한 인증 방식에 따라 적절한 provider를 생성한다.
    if settings.auth_provider == "refresh":
        return RefreshTokenProvider(settings)
    if settings.auth_provider == "env":
        return EnvTokenProvider(settings.access_token)
    if settings.auth_provider == "selenium":
        return SeleniumTokenProvider(settings)
    raise RuntimeError(f"Unsupported CG_AUTH_PROVIDER: {settings.auth_provider}")
