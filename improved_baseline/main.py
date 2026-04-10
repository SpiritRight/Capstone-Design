from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import ffmpeg
import numpy as np
from sseclient import SSEClient
from ultralytics import YOLO

try:
    from .auth import AuthorizedSession, build_token_provider
    from .config import Settings, load_settings
except ImportError:
    from auth import AuthorizedSession, build_token_provider
    from config import Settings, load_settings


LOGGER = logging.getLogger("improved_baseline")


@dataclass
class TrashCandidate:
    # 한 번 감지된 쓰레기 후보가 같은 자리에서 유지되는지 추적한다.
    bbox: list[float]
    last_seen: int
    stationary_count: int = 1
    illegal: bool = False
    saved: bool = False
    initial_area: float = 0.0

    @classmethod
    def create(cls, bbox: list[float], frame_number: int) -> "TrashCandidate":
        return cls(
            bbox=bbox,
            last_seen=frame_number,
            initial_area=bbox[2] * bbox[3],
        )

    def update(self, new_bbox: list[float], frame_number: int, settings: Settings) -> None:
        cx0 = self.bbox[0] + self.bbox[2] / 2
        cy0 = self.bbox[1] + self.bbox[3] / 2
        cx1 = new_bbox[0] + new_bbox[2] / 2
        cy1 = new_bbox[1] + new_bbox[3] / 2
        dist = np.hypot(cx1 - cx0, cy1 - cy0)
        area = new_bbox[2] * new_bbox[3]

        if area > self.initial_area * settings.area_growth_ratio:
            self.stationary_count = 1
            self.initial_area = area
        else:
            self.stationary_count += 1 if dist < settings.movement_threshold else -self.stationary_count + 1

        self.bbox = new_bbox
        self.last_seen = frame_number
        if self.stationary_count >= settings.stationary_frames_threshold:
            self.illegal = True


def build_session(settings: Settings) -> AuthorizedSession:
    # worker와 poller가 동일한 인증 방식을 쓰도록 session 생성 로직을 통일한다.
    return AuthorizedSession(build_token_provider(settings))


def encode_batch_with_ffmpeg(frames: list[np.ndarray], out_path: str, fps: int) -> None:
    # 이벤트 전후 프레임을 하나의 mp4 스니펫으로 인코딩한다.
    if not frames:
        return

    height, width, _ = frames[0].shape
    process = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="bgr24", s=f"{width}x{height}", framerate=fps)
        .output(
            out_path,
            vcodec="libx264",
            pix_fmt="yuv420p",
            crf=23,
            preset="veryfast",
            movflags="+faststart",
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    for frame in frames:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def send_video_to_server(filepath: str, stream_id: str, settings: Settings) -> None:
    # 탐지 이벤트 영상을 서버로 업로드한다.
    session = build_session(settings)
    with open(filepath, "rb") as handle:
        files = {
            "image": handle,
            "timestamp": (None, datetime.now().isoformat()),
            "stream": (None, stream_id),
        }
        response = session.post(
            settings.upload_url,
            files=files,
            timeout=settings.request_timeout_seconds,
        )
        response.raise_for_status()
        LOGGER.info("[%s] uploaded %s -> %s", stream_id, os.path.basename(filepath), response.status_code)


def config_poller(queue: mp.Queue, settings: Settings) -> None:
    # SSE로 활성 카메라 목록을 받아 신규 스트림만 worker queue에 넣는다.
    session = build_session(settings)
    seen: dict[str, datetime] = {}
    stale_timeout = timedelta(seconds=settings.stale_timeout_seconds)

    while True:
        now = datetime.utcnow()
        for stream_id, last_seen in list(seen.items()):
            if now - last_seen > stale_timeout:
                del seen[stream_id]

        try:
            response = session.get(
                settings.config_url,
                timeout=(settings.sse_connect_timeout_seconds, settings.sse_read_timeout_seconds),
                stream=True,
            )
            response.raise_for_status()
            client = SSEClient(response)

            for event in client.events():
                if not event.data:
                    continue

                payload = json.loads(event.data)
                items = payload if isinstance(payload, list) else [payload]
                for cfg in items:
                    if not cfg.get("id") or not cfg.get("passwd"):
                        continue
                    stream_id = cfg["stream"]
                    if stream_id not in seen:
                        queue.put(cfg)
                        LOGGER.info("[poller] queued stream %s", stream_id)
                    seen[stream_id] = datetime.utcnow()
        except Exception:
            LOGGER.exception("[poller] failed to receive camera inventory")

        time.sleep(settings.poll_retry_seconds)


def camera_worker(cfg: dict, settings: Settings) -> None:
    # 카메라별 독립 프로세스: RTSP 수신, YOLO 추론, 이벤트 저장/업로드를 담당한다.
    stream_id = cfg["stream"]
    ip = cfg["ip"]
    user = cfg["id"]
    password = cfg["passwd"]
    worker_logger = logging.getLogger(f"improved_baseline.{stream_id}")

    base = os.path.join(settings.capture_root, stream_id)
    frames_dir = os.path.join(base, "frames")
    video_dir = os.path.join(base, "video")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    rtsp_url = f"rtsp://{user}:{password}@{ip}:{settings.rtsp_port}"
    ffmpeg_cmd = [
        "ffmpeg",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-an",
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(settings.fps),
        "-f",
        "rawvideo",
        "-",
    ]
    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
    )

    model = YOLO(settings.yolo_model_path)
    previous_frames = deque(maxlen=settings.pre_frames)
    events: list[dict] = []
    trash_candidates: list[TrashCandidate] = []
    frame_number = 0
    frame_bytes = settings.frame_width * settings.frame_height * 3

    worker_logger.info("worker started (pid=%s)", os.getpid())

    def update_trash(detections: list[list[float]], current_frame_number: int) -> list[TrashCandidate]:
        # 현재 프레임의 탐지 결과를 기존 후보와 매칭해 무단투기 후보를 갱신한다.
        nonlocal trash_candidates

        for detection in detections:
            cx = detection[0] + detection[2] / 2
            cy = detection[1] + detection[3] / 2
            match = next(
                (
                    candidate
                    for candidate in trash_candidates
                    if np.hypot(
                        (candidate.bbox[0] + candidate.bbox[2] / 2) - cx,
                        (candidate.bbox[1] + candidate.bbox[3] / 2) - cy,
                    )
                    < settings.match_distance_threshold
                ),
                None,
            )

            if match is None:
                # 가까운 기존 후보가 없으면 새 쓰레기 후보로 등록한다.
                trash_candidates.append(TrashCandidate.create(detection, current_frame_number))
            else:
                # 기존 후보와 충분히 가깝다면 같은 객체로 보고 정지 여부를 누적 계산한다.
                match.update(detection, current_frame_number, settings)

        trash_candidates = [
            candidate
            for candidate in trash_candidates
            if current_frame_number - candidate.last_seen
            < settings.stationary_frames_threshold + settings.max_missing_frames
        ]

        new_illegal = []
        for candidate in trash_candidates:
            if candidate.illegal and not candidate.saved:
                # 이미 이벤트를 만든 후보는 다시 저장하지 않도록 saved 플래그를 둔다.
                candidate.saved = True
                new_illegal.append(candidate)
        return new_illegal

    try:
        while True:
            # ffmpeg stdout에서 raw frame을 직접 읽어온다.
            raw = process.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                worker_logger.warning("stream ended")
                break

            frame = np.frombuffer(raw, np.uint8).reshape(
                (settings.frame_height, settings.frame_width, 3)
            ).copy()
            frame_number += 1

            preview = cv2.resize(frame, (settings.save_width, settings.save_height))
            preview_path = os.path.join(frames_dir, f"frame_{frame_number:06d}.jpg")
            cv2.imwrite(preview_path, preview)

            # 저장 프레임 수가 너무 커지지 않게 오래된 이미지부터 정리한다.
            frame_files = sorted(os.listdir(frames_dir))
            if len(frame_files) > settings.max_stored_frames:
                excess = len(frame_files) - settings.max_stored_frames
                for old_file in frame_files[:excess]:
                    try:
                        os.remove(os.path.join(frames_dir, old_file))
                    except OSError:
                        continue

            detections = []
            # YOLO 추론 결과를 bbox 리스트로 변환한다.
            results = model.predict(source=frame, conf=settings.detection_confidence, verbose=False)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append([x1, y1, x2 - x1, y2 - y1])

            for _candidate in update_trash(detections, frame_number):
                # 불법 투기 이벤트가 처음 확정되면 이전 프레임과 현재 프레임을 보관한다.
                events.append(
                    {
                        "prevs": [prev.copy() for prev in previous_frames],
                        "event_frame": frame.copy(),
                        "posts": [],
                        "count": 0,
                        "pending": True,
                    }
                )

            for event in events[:]:
                if event["pending"]:
                    # 이벤트가 방금 생성된 프레임은 posts에 중복 저장하지 않기 위해 한 번 건너뛴다.
                    event["pending"] = False
                    continue

                # 이벤트 발생 후 일정 개수의 후행 프레임을 모아 영상 스니펫을 완성한다.
                if event["count"] < settings.post_frames:
                    event["posts"].append(frame.copy())
                    event["count"] += 1

                if event["count"] >= settings.post_frames:
                    # 이전 프레임 + 이벤트 프레임 + 이후 프레임을 합쳐 최종 클립을 만든다.
                    snippet = event["prevs"] + [event["event_frame"]] + event["posts"]
                    video_name = f"event_{frame_number:06d}.mp4"
                    video_path = os.path.join(video_dir, video_name)
                    encode_batch_with_ffmpeg(snippet, video_path, settings.fps)
                    worker_logger.info("snippet saved: %s", video_name)
                    try:
                        send_video_to_server(video_path, stream_id, settings)
                    except Exception:
                        worker_logger.exception("upload failed for %s", video_name)
                    events.remove(event)

            previous_frames.append(frame.copy())
    finally:
        process.terminate()
        process.wait()
        worker_logger.info("worker exiting")


def main() -> None:
    # 메인 프로세스는 인증 확인, poller 시작, worker 프로세스 관리만 담당한다.
    settings = load_settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        build_token_provider(settings).get_token()
    except Exception:
        LOGGER.exception("initial auth bootstrap failed")
        raise

    queue: mp.Queue = mp.Queue()
    poller = mp.Process(target=config_poller, args=(queue, settings), daemon=True)
    poller.start()
    LOGGER.info("poller started (pid=%s)", poller.pid)
    workers: dict[str, mp.Process] = {}

    try:
        while True:
            cfg = queue.get()
            stream_id = cfg["stream"]
            existing = workers.get(stream_id)
            if existing is not None and existing.is_alive():
                continue

            process = mp.Process(target=camera_worker, args=(cfg, settings), daemon=False)
            process.start()
            workers[stream_id] = process
            LOGGER.info("worker started for %s (pid=%s)", stream_id, process.pid)
            time.sleep(1)
    except KeyboardInterrupt:
        LOGGER.info("shutting down")
        for process in workers.values():
            if process.is_alive():
                os.kill(process.pid, signal.SIGTERM)
        poller.terminate()


if __name__ == "__main__":
    main()
