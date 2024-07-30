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
        face = image[y:y+h, x:x+w]
        face = cv2.GaussianBlur(face, (99, 99), 30)
        image[y:y+h, x:x+w] = face
    return image

def process_video(input_video_path, output_video_path, model):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_faces(frame, model)
        frame = blur_faces(frame, boxes)
        out.write(frame)

    cap.release()
    out.release()
