import cv2
import torch
import time
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np

# Load BLIP model and processor
print("Loading processor")
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base", 
    use_fast=True
    )
print("Loading model")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Device selection for Mac: prefer MPS (Apple Silicon), then CUDA, then CPU
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon Metal backend (mps).")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA GPU.")
else:
    device = "cpu"
    print("Using CPU.")

#device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Start the webcam
capture = cv2.VideoCapture(0)

print("Press 'q' to quit.")

prev_time = time.time()
fps = 0

while True:
    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame (numpy array) to PIL Image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    # Preprocess and predict
    inputs = processor(images=pil_img, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(**inputs)
    
    description = processor.decode(out[0], skip_special_tokens=True)    
    print("Description:", description)

    # Display the frame with caption
    cv2.putText(frame, description, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Real-Time Image Captioning', frame)

    # Wait for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()