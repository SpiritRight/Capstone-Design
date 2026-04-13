from ultralytics import YOLO
import wandb

# ✅ WandB 로그인
wandb.login(key="")

# ✅ WandB 프로젝트 초기화
wandb.init(
    project="Train Garbage_Bag",
    name="쓰레기봉투 데이터셋",
    mode="online",
    config={
        "model": "yolo11m.pt",
        "epochs": 50,
        "batch": 2,  
        "imgsz": 1280,  
        "copy_paste": 0.5,  
        "mosaic": 0.4, 
        "mixup": 0.1,
        # "lr0": 0.005, 
        # "lrf": 0.001,
        "iou": 0.45,
        "conf": 0.3,
        # "cos_lr":True,
        # "single_cls":True,
        "erasing" : 0.2,
    }
)

# ✅ YOLO 모델 불러오기
# model = YOLO("/mnt/d/runs/detect/train28/weights/best.pt")
model = YOLO("yolo11m.pt")

# ✅ 학습 진행
model.train(
    data="/mnt/d/캡스톤/datasets/data_original.yaml",
    epochs=50,
    imgsz=1280,
    batch=4,
    copy_paste=0.5,
    erasing=0.2,
    # mixup=0.25,
    # lr0=0.005,  
    # lrf=0.001,
    # patience=10,
    rect=True,
    # cos_lr=True,
    conf=0.3,
    iou=0.45,
    val=True,
    plots=True,
    mosaic = 0.4,
    mixup = 0.1,
    # single_cls = True,
    # resume=Trupip
    
)

# ✅ WandB 종료
wandb.finish()
