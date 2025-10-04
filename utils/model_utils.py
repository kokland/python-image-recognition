import torch
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

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

def get_caption(processor, model, device, frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description    