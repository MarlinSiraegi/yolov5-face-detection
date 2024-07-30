import cv2
import numpy as np

def detect_faces(image, model):
    results = model(image)
    boxes = []
    for *xyxy, conf, cls in results.xyxy[0]:  # x1, y1, x2, y2, confidence, class
        if conf > 0.4:  # confidence threshold
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append((x1, y1, x2-x1, y2-y1))
    return boxes

def blur_faces(image, boxes):
    for (x, y, w, h) in boxes:
        if x < 0 or y < 0 or x+w > image.shape[1] or y+h > image.shape[0]:
            continue
        face = image[y:y+h, x:x+w]
        if face.size != 0:
            face = cv2.GaussianBlur(face, (99, 99), 30)
            image[y:y+h, x:x+w] = face
    return image

def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is not None and height is not None:
        ratio = min(width / float(w), height / float(h))
        dim = (int(w * ratio), int(h * ratio))
    elif width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def process_video(input_video_path, output_video_path, model, output_width=None, output_height=None):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video FPS: {fps}, Original Width: {original_width}, Original Height: {original_height}")

    if output_width is None and output_height is None:
        output_width = original_width
        output_height = original_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 입력 프레임의 크기 조정 (비율 유지)
        frame_resized = resize_with_aspect_ratio(frame, width=output_width, height=output_height)

        # 크기 조정된 프레임의 비율을 유지하며 출력 크기로 맞추기 위해 패딩 추가
        delta_w = output_width - frame_resized.shape[1]
        delta_h = output_height - frame_resized.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        color = [0, 0, 0]
        frame_padded = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        boxes = detect_faces(frame_padded, model)
        frame_blurred = blur_faces(frame_padded, boxes)

        out.write(frame_blurred)

    cap.release()
    out.release()
    print(f"Processed video saved as {output_video_path}")