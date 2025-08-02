import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "/path/to/your/model_checkpoint.pth"
