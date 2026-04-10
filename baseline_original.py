#!/usr/bin/env python3
import os
import signal
import time
import logging
import json
import requests
import subprocess
import multiprocessing as mp
import ffmpeg                       # ★ ffmpeg-python
import sys
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from collections import deque

from datetime import datetime, timedelta

import cv2
import numpy as np
from sseclient import SSEClient
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from ultralytics import YOLO

# --- 전역 설정 ---
CONFIG_URL   = "http://3.36.174.53:8080/cleanguard/cctv/sse"
LOGIN_URL    = "http://3.36.174.53:8080/oauth2/authorization/kakao"
AWS_API_URL  = "http://3.36.174.53:8080/cleanguard/image/"
CAPTURE_ROOT = "captures"

# 무단투기 감지 파라미터
MOVEMENT_THRESHOLD          = 10
STATIONARY_FRAMES_THRESHOLD = 5
MATCH_DISTANCE_THRESHOLD    = 30
MAX_STORED_FRAMES = 1000

# 이벤트 스니펫 설정
FPS         = 5       # 1FPS
PRE_FRAMES  = 35       # 이전 5프레임
POST_FRAMES = 35       # 이후 5프레임

# --------------------------------------------------------------------
# OAuth 토큰 얻기 (셀레니움)
def get_access_token():
    pid = os.getpid()
    svc = Service(ChromeDriverManager().install(), log_path=f"chromedriver_{pid}.log")
    opts = webdriver.ChromeOptions()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=svc, options=opts)
    driver.get(LOGIN_URL)
    driver.find_element(By.NAME, "loginId").send_keys("<LOGIN_ID>")
    driver.find_element(By.NAME, "password").send_keys("<LOGIN_PASSWORD>", Keys.RETURN)
    time.sleep(5)
    url = driver.current_url
    driver.quit()
    qs = parse_qs(urlparse(url).query)
    return qs.get("access_token", [None])[0]

# --------------------------------------------------------------------
# SSE로 활성 CCTV 리스트를 폴링해 큐에 넣기

def config_poller(q: mp.Queue, bearer_token: str):
    headers = {"Authorization": f"Bearer {bearer_token}"}
    # sid → 마지막 감지 시각
    seen = {}
    STALE_TIMEOUT = timedelta(seconds=60)

    while True:
        now = datetime.utcnow()
        # 1) 일정 시간 지나면 seen에서 제거
        for sid, last_seen in list(seen.items()):
            if now - last_seen > STALE_TIMEOUT:
                del seen[sid]

        try:
            resp   = requests.get(CONFIG_URL, headers=headers, timeout=5, stream=True)
            client = SSEClient(resp)
            for event in client.events():
                if not event.data:
                    continue
                raw   = json.loads(event.data)
                items = raw if isinstance(raw, list) else [raw]

                for cfg in items:
                    if not cfg.get("id") or not cfg.get("passwd"):
                        continue
                    sid = cfg["stream"]

                    # 2) seen에 없으면 신규 또는 복귀 스트림으로 간주
                    if sid not in seen:
                        q.put(cfg)
                        print(f"[poller] queued stream {sid}")

                    # 3) 마지막 감지 시각 갱신
                    seen[sid] = now

        except Exception as e:
            print(f"[poller] error: {e}")

        time.sleep(5)




def encode_batch_with_ffmpeg(frames, out_path: str, fps: int):
    if not frames:
        return
    h, w, _ = frames[0].shape
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24',
               s=f'{w}x{h}', framerate=fps)
        .output(out_path,
                vcodec='libx264', pix_fmt='yuv420p',
                crf=23, preset='veryfast',
                movflags='+faststart')
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )
    for f in frames:
        process.stdin.write(f.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()

# --------------------------------------------------------------------
# MP4 생성→AWS 전송
def send_video_to_aws(filepath: str, stream_id: str, headers: dict):
    with open(filepath, "rb") as f:
        files = {
            "image": f,
            "timestamp": (None, datetime.now().isoformat()),
            "stream":   (None, stream_id),
        }
        r = requests.post(AWS_API_URL, files=files, headers=headers)
        print(f"[{stream_id}] uploaded {os.path.basename(filepath)} -> {r.status_code}")

# --------------------------------------------------------------------
# 2) 카메라 워커: RAW→프레임 저장→무단투기 감지 이벤트 스니펫
def camera_worker(cfg: dict, bearer_token: str):
    stream_id, ip, user, pw = cfg["stream"], cfg["ip"], cfg["id"], cfg["passwd"]
    print(f"[{stream_id}] Worker {os.getpid()} start")

    base       = os.path.join(CAPTURE_ROOT, stream_id)
    frames_dir = os.path.join(base, "frames")
    video_dir  = os.path.join(base, "video")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(video_dir,  exist_ok=True)

    headers = {"Authorization": f"Bearer {bearer_token}"}

    # FFmpeg → RAW BGR24
    rtsp_url = f"rtsp://{user}:{pw}@{ip}:554"
    width, height = 2304, 1296
    ffmpeg_cmd = [
        "ffmpeg", "-rtsp_transport", "tcp", "-i", rtsp_url,
        "-an", "-c:v", "rawvideo", "-pix_fmt", "bgr24",
        "-r", str(FPS), "-f", "rawvideo", "-"
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

    model = YOLO("yolo11m_train.engine")
    prev_deque = deque(maxlen=PRE_FRAMES)
    events = []
    trash_candidates = []

    class TrashCandidate:
        def __init__(self, bbox, fn):
            self.bbox = bbox
            self.last_seen = fn
            self.stationary_count = 1
            self.illegal = False
            self.saved = False
            self.initial_area = bbox[2]*bbox[3]
        def update(self, new_bbox, fn):
            cx0, cy0 = self.bbox[0]+self.bbox[2]/2, self.bbox[1]+self.bbox[3]/2
            cx1, cy1 = new_bbox[0]+new_bbox[2]/2, new_bbox[1]+new_bbox[3]/2
            dist = np.hypot(cx1-cx0, cy1-cy0)
            area = new_bbox[2]*new_bbox[3]
            if area > self.initial_area * 1.1:
                self.stationary_count = 1
                self.initial_area = area
            else:
                self.stationary_count += 1 if dist < MOVEMENT_THRESHOLD else -self.stationary_count+1
            self.bbox = new_bbox
            self.last_seen = fn
            if self.stationary_count >= STATIONARY_FRAMES_THRESHOLD:
                self.illegal = True

    def update_trash(dets, fn):
        nonlocal trash_candidates
        for d in dets:
            cx, cy = d[0]+d[2]/2, d[1]+d[3]/2
            match = next((c for c in trash_candidates
                          if np.hypot((c.bbox[0]+c.bbox[2]/2)-cx,
                                      (c.bbox[1]+c.bbox[3]/2)-cy) < MATCH_DISTANCE_THRESHOLD), None)
            if match:
                match.update(d, fn)
            else:
                trash_candidates.append(TrashCandidate(d, fn))
        trash_candidates = [c for c in trash_candidates if fn - c.last_seen < STATIONARY_FRAMES_THRESHOLD + 10]

    fn = 0
    while True:
        raw = proc.stdout.read(width * height * 3)
        if len(raw) < width * height * 3:
            print(f"[{stream_id}] stream ended")
            break
        frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3)).copy()
        fn += 1

        # 1) 저장 (640×480)
        save_frame = cv2.resize(frame, (640, 480))
        frame_path = os.path.join(frames_dir, f"frame_{fn:06d}.jpg")
        cv2.imwrite(frame_path, save_frame)

        # 2) MAX_STORED_FRAMES 초과 시 '가장 오래된' 파일부터 모두 정리
        files = sorted(os.listdir(frames_dir))
        if len(files) > MAX_STORED_FRAMES:
            # 초과 개수만큼 맨 앞(가장 오래된)부터 삭제
            excess = len(files) - MAX_STORED_FRAMES
            for old_file in files[:excess]:
                try:
                    os.remove(os.path.join(frames_dir, old_file))
                except OSError:
                    pass

        # 2) 탐지
        dets = []
        res = model.predict(source=frame, conf=0.5)
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                dets.append([x1,y1,x2-x1,y2-y1])
                cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),2)

        update_trash(dets, fn)
        prevs = list(prev_deque)
        for c in trash_candidates:
            if c.illegal and not c.saved:
                c.saved = True
                events.append({"prevs": prevs, "event_frame": frame, "posts": [], "count": 0, "pending": True})

        for ev in events[:]:
            if ev["pending"]:
                ev["pending"] = False
            else:
                if ev["count"] < POST_FRAMES:
                    ev["posts"].append(frame)
                    ev["count"] += 1
                if ev["count"] >= POST_FRAMES:
                    snippet = ev["prevs"] + [ev["event_frame"]] + ev["posts"]
                    vid_name = f"event_{fn:06d}.mp4"
                    vid_path = os.path.join(video_dir, vid_name)
                    encode_batch_with_ffmpeg(snippet, vid_path, FPS)

                    print(f"[{stream_id}] snippet saved: {vid_name}")
                    send_video_to_aws(vid_path, stream_id, headers)
                    events.remove(ev)

        prev_deque.append(frame)

    proc.terminate()
    proc.wait()
    print(f"[{stream_id}] Worker exiting")

# --------------------------------------------------------------------
# 3) 메인: 토큰 획득 → 폴링 → 워커 관리
def main():
    logging.basicConfig(level=logging.INFO)
    mp.set_start_method("spawn")

    token = get_access_token()
    if not token:
        print("❌ OAuth 토큰 실패, 종료")
        return

    cfg_q = mp.Queue()
    poller = mp.Process(target=config_poller, args=(cfg_q, token), daemon=True)
    poller.start()
    workers = {}

    try:
        while True:
            cfg = cfg_q.get()
            sid = cfg["stream"]
            if sid not in workers:
                p = mp.Process(target=camera_worker, args=(cfg, token), daemon=False)
                p.start()
                workers[sid] = p
                print(f"[main] started worker for {sid} (pid {p.pid})")
            time.sleep(1)
    except KeyboardInterrupt:
        print("[main] shutting down…")
        for p in workers.values(): os.kill(p.pid, signal.SIGTERM)
        poller.terminate()

if __name__ == "__main__":
    main()
