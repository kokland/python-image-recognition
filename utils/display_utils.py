import cv2
import time

def draw_caption(frame, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 255, 0)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    frame_height, frame_width = frame.shape[:2]
    x = (frame_width - text_width) // 2
    y = frame_height - 20  # 20px from bottom
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_fps(frame, fps):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (255, 0, 0)
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)

def calculate_fps(prev_time, new_time):
    fps = 1 / (new_time - prev_time)
    prev_time = new_time
    return fps