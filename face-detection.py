# import cv2
# import torch
# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image
# import numpy as np
# import time
# from model_utils import load_processor, load_model, get_device, get_caption
# from face_utils import load_face_cascade, draw_rectangle_on_faces
# from display_utils import draw_caption, draw_fps
# from camera_utils import set_camera_resolution

from utils.camera_utils import set_camera_resolution
from utils.display_utils import draw_caption, draw_fps
from utils.face_utils import load_face_cascade, draw_rectangle_on_faces
from utils.model_utils import load_processor, load_model, get_device, get_caption
import cv2
import time


def main():
    device = get_device()
    processor = load_processor()
    model = load_model(device)
    face_cascade = load_face_cascade()
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