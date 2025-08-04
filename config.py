import os
import torch
import urllib.request

device = torch.device("cpu")

# === URLs for model and tokenizer ===
model_url = "https://drive.google.com/uc?export=download&id=1Zv_xZNgm-3rg97CeFgU3sByKVroS0EDz"
tokenizer_url = "https://drive.google.com/uc?export=download&id=1VYs3TJh5p7dLHC5OuVjAYlykFkli379e"

checkpoint_path = "ocr_checkpoint.pt"
tokenizer_path = "tokenizer.pkl"

# === Download model checkpoint if needed ===
if not os.path.exists(checkpoint_path):
    print("Downloading model checkpoint")
    urllib.request.urlretrieve(model_url, checkpoint_path)
    print("Model downloaded")

# === Download tokenizer if needed ===
if not os.path.exists(tokenizer_path):
    print("Downloading tokenizer")
    urllib.request.urlretrieve(tokenizer_url, tokenizer_path)
    print("Tokenizer downloaded")
