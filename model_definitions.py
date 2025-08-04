# model_definitions.py
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

max_height, max_width = 384, 512

transform = transforms.Compose([
    transforms.Resize((max_height, max_width)),
    transforms.ToTensor()
])

class LatexTokenizer:
    def __init__(self):
        self.specials = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        self.vocab = self.specials.copy()
        self.t2i = {tok: i for i, tok in enumerate(self.vocab)}

    def build_vocab(self, texts):
        toks = set(tok for txt in texts for tok in txt.split())
        self.vocab = self.specials + sorted(toks)
        self.t2i = {tok: i for i, tok in enumerate(self.vocab)}

    def encode(self, txt):
        return [self.t2i['<SOS>']] + [self.t2i.get(tok, self.t2i['<UNK>']) for tok in txt.split()] + [self.t2i['<EOS>']]

    def decode(self, ids):
        return ' '.join(self.vocab[i] for i in ids if self.vocab[i] not in self.specials)

import pickle

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

class OCRModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
        )
        conv_out = 64 * (max_height // 4) * (max_width // 4)
        self.fc = nn.Linear(conv_out, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=4,
            num_encoder_layers=3, num_decoder_layers=3
        )
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, imgs, tgt, tgt_mask=None):
        B, _, h, w = imgs.shape
        enc = self.encoder(imgs).view(B, -1)
        enc = self.fc(enc).unsqueeze(0)
        tgt_emb = self.embedding(tgt)
        out = self.transformer(enc, tgt_emb, tgt_mask=tgt_mask)
        return self.out(out)

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

def predict(img, model, vocab, max_len=60):
    model.eval()
    if isinstance(img, torch.Tensor):
        img = img.unsqueeze(0)
    else:
        img = transform(img).unsqueeze(0)

    seq = torch.tensor([[vocab['<SOS>']]])
    for _ in range(max_len):
        tgt_mask = generate_square_subsequent_mask(seq.size(0))
        logits = model(img, seq, tgt_mask=tgt_mask)
        next_token = logits.argmax(-1)[-1, 0].item()
        seq = torch.cat([seq, torch.tensor([[next_token]])], dim=0)
        if next_token == vocab['<EOS>']:
            break

    return tokenizer.decode(seq.squeeze().tolist())

print("Tokenizer vocab size:", len(tokenizer.vocab))