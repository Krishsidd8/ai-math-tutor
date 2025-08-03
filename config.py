import os
import torch
import urllib.request

device = torch.device("cpu")

model_url = "https://drive.google.com/uc?export=download&id=1IttUFMaSxgyEbunjwvnOntFtcWWWuQdh"
checkpoint_path = "ocr_checkpoint.pt"

if not os.path.exists(checkpoint_path):
    print("Downloading model checkpoint...")
    urllib.request.urlretrieve(model_url, checkpoint_path)
    print("Model downloaded successfully.")