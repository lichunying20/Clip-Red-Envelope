# lichunying
## clip原理
在训练阶段，对于一个batch 的数据，首先通过文本编码器和图像编码器，得到文本和图像的特征，接着将所有的文本和图像特征分别计算内积，就能得到一个矩阵，然后从图像的角度看，行方向就是一个分类器，从文本角度看，列方向也是一个分类器。而由于我们已经知道一个batch中的文本和图像的匹配关系，所以目标函数就是最大化同一对图像和文本特征的内积，也就是矩阵对角线上的元素，而最小化与不相关特征的内积。

在测试阶段，可以直接将训练好的CLIP用于其他数据集而不需要finetune。和训练阶段类似，首先将需要分类的图像经过编码器得到特征，然后对于目标任务数据集的每一个标签，或者你自己定义的标签，都构造一段对应的文本，如上图中的 dog 会改造成 “A photo of a dog”，以此类推。然后经过编码器得到文本和图像特征，接着将文本特征与图像特征做内积，内积最大对应的标签就是图像的分类结果。

## 所用图片
![红包1](https://user-images.githubusercontent.com/128216499/227192980-ffeab773-c45e-4ddb-8960-b88a1d0703c2.jpeg)

## clip代码及最后结果
### 当classes = ['a red packet', 'a dog', 'a cat']时
```python
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
  
print('classes:{} score:{:.2f}'.format(classes[indices.item()], values.item()))
```

### 最后结果
![image](https://user-images.githubusercontent.com/128216499/227191828-b7e41bb6-990a-41d1-a91c-383d36881742.png)

### 当list=('red','envelope','China')时
```python
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("红包1.jpeg")).unsqueeze(0).to(device)
list = ('red','envelope','China')
text = clip.tokenize([( f"a photo of {c}.") for c in list]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    similarity = (image_features @ text_features.T).softmax(dim=-1)  # 对图像描述和图像特征
    values, indices = similarity[0].topk(3)

print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{list[index]:>16s}: {100 * value.item():.2f}%")
```

### 最后结果
![image](https://user-images.githubusercontent.com/128216499/227190653-078240d0-05e3-4424-bd12-d4bdb11e20cb.png)



### 当list=('red','envelope','China','Red Envelope')时
```python
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("红包1.jpeg")).unsqueeze(0).to(device)
list = ('red','envelope','China','Red Envelope')
text = clip.tokenize([( f"a photo of {c}.") for c in list]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    similarity = (image_features @ text_features.T).softmax(dim=-1)  # 对图像描述和图像特征
    values, indices = similarity[0].topk(4)

print("\nTop predictions:\n")
for value, index in zip(values, indices):
    print(f"{list[index]:>16s}: {100 * value.item():.2f}%")
```



### 最后结果
![image](https://user-images.githubusercontent.com/128216499/227189841-bf4195cd-ae49-4ea1-a70d-ab1dec57e4de.png)
