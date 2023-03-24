import torch
import clip
from PIL import Image


classes = ['a red packet', 'a dog', 'a cat']

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("红包1.jpeg")).unsqueeze(0).to(device)
text = clip.tokenize(["a red packet",  "a dog", "a cat"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # 对图像描述和图像特征
    values, indices = similarity[0].topk(1)
    #logits_per_image, logits_per_text = model(image, text)
    #probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
print('classes:{} score:{:.2f}'.format(classes[indices.item()], values.item()))
