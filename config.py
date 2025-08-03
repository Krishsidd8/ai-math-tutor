import os
import torch
import gdown

device = torch.device("cpu")
checkpoint_path = "ocr_checkpoint.pt"

if not os.path.exists(checkpoint_path):
    print("Downloading model checkpoint from Google Drive...")
    gdown.download(
        "https://drive.google.com/uc?id=1IttUFMaSxgyEbunjwvnOntFtcWWWuQdh",
        checkpoint_path,
        quiet=False
    )
    print("Model downloaded successfully.")
