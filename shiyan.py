import os
import clip
import torch
from torchvision.datasets import CIFAR100
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("红包1.jpeg")).unsqueeze(0).to(device)
list = ('red','envelope','China','Red Envelope')
#list = ('red','envelope','China')
text = clip.tokenize([( f"a photo of {c}.") for c in list]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    similarity = (image_features @ text_features.T).softmax(dim=-1)  # 对图像描述和图像特征
    values, indices = similarity[0].topk(4)

print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{list[index]:>16s}: {100 * value.item():.2f}%")