import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import time

def load_processor():
    print("Loading processor")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    return processor

def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon Metal backend (mps).")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU.")
    else:
        device = "cpu"
        print("Using CPU.")
    return device

def load_model(device):
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")    
    model.to(device)
    return model

def set_camera_resolution(cap, width=640, height=480):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def get_caption(processor, model, device, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

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

def load_face_cascade():
    print("Loading face cascade")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

def draw_rectangle_on_faces(frame, face_cascade):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    print(f"Detected {len(faces)} faces")

    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)        
    

def main():
    device = get_device()
    processor = load_processor()
    model = load_model(device)
    face_cascade = load_face_cascade()
    #processor, model, device = load_model_and_processor()
    cap = cv2.VideoCapture(0)
    set_camera_resolution(cap, 1024, 768)

    prev_time = time.time()
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Get caption for frame (may be slow)
        draw_rectangle_on_faces(frame, face_cascade)
        description = get_caption(processor, model, device, frame)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        # Draw FPS and caption
        draw_fps(frame, fps)
        draw_caption(frame, description)

        cv2.imshow('Real-Time Image Captioning', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()