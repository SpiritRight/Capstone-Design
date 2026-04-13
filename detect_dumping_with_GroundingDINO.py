import os
import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate

# 모델 로드
model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py", "weights/groundingdino_swinb_cogcoor.pth")

# 이미지 폴더 경로 설정
# IMAGE_FOLDER = "/mnt/d/캡스톤/dataset/valid/images"
# OUTPUT_FOLDER = "/mnt/d/캡스톤/datasets/val/images_garbagebag"
# YOLO_LABELS_FOLDER = "/mnt/d/캡스톤/datasets/val/labels_garbagebag"

IMAGE_FOLDER = "/mnt/d/캡스톤/datasets/val/images"
OUTPUT_FOLDER = "/mnt/d/캡스톤/datasets/result/images_holding_bag"
YOLO_LABELS_FOLDER = "/mnt/d/캡스톤/datasets/result/labels_holding_bag"

TEXT_PROMPT = "person with trash bag ."
BOX_TRESHOLD = 0.4
TEXT_TRESHOLD = 0.8

# 결과 저장 폴더 생성
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(YOLO_LABELS_FOLDER, exist_ok=True)

# 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 전체 이미지 개수
total_images = len(image_files)

# 이미지 처리 루프
for idx, image_name in enumerate(image_files, start=1):
    image_path = os.path.join(IMAGE_FOLDER, image_name)

    print(f"Processing image {idx}/{total_images}: {image_name}")

    # 이미지 로드
    image_source, image = load_image(image_path)

    # 이미지 크기 가져오기
    H, W, _ = image_source.shape
    # print('H = ', H)
    # print('W = ', W)
    # 객체 검출 수행
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    # Annotated 이미지 생성
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # 결과 이미지 저장
    output_path = os.path.join(OUTPUT_FOLDER, f"annotated_{image_name}")
    cv2.imwrite(output_path, annotated_frame)


    # YOLO 라벨 저장 (.txt 파일)
    label_path = os.path.join(YOLO_LABELS_FOLDER, f"{os.path.splitext(image_name)[0]}.txt")

    with open(label_path, "w") as f:
        for box in boxes:
            # print(f"Detected box: {box}")  # 🔥 박스 좌표 출력 (디버깅용)
            
            x_min, y_min, width, height = box  # ✅ GroundingDINO 형식 그대로 사용

            # 🔥 YOLO 좌표 변환
            # x_center = x_min + (width / 2.0)  # ✅ YOLO 형식 변환
            # y_center = y_min + (height / 2.0)  # ✅ YOLO 형식 변환


            # ✅ YOLO 포맷: "class x_center y_center width height"
            f.write(f"1 {x_min:.6f} {y_min:.6f} {width:.6f} {height:.6f}\n")



print("\n✅ 모든 이미지 처리가 완료되었습니다!")
