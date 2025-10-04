import cv2

def set_camera_resolution(cap, width=640, height=480):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def get_video_capture_device(index=0):
    print(f"Accessing video capture device at index {index}")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video capture device at index {index}")
    return cap