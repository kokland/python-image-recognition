from utils.camera_utils import set_camera_resolution, get_video_capture_device
from utils.display_utils import draw_caption, draw_fps, calculate_fps
from utils.face_utils import load_face_cascade, draw_rectangle_on_faces
from utils.model_utils import load_processor, load_model, get_device, get_caption
import cv2
import time


def main():
    device = get_device()
    processor = load_processor()
    model = load_model(device)
    face_cascade = load_face_cascade()
    cap = get_video_capture_device(0)
    
    cap = set_camera_resolution(cap, 800, 480)

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
        temp_time = time.time()
        fps = calculate_fps(prev_time, temp_time)
        prev_time = temp_time

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