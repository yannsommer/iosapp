import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 准备图像
image_path = "IMG_4342.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# 准备文本
texts = ["prettiest girl", "a cat"]
text = clip.tokenize(texts).to(device)

# 计算相似度
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 打印结果
for i, label in enumerate(["prettiest girl", "a cat"]):
    print(f"{label}: {probs[0][i]:.2%}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


# 显示图像
img = Image.open(image_path)
ax1.imshow(img)
ax1.axis('off')
ax1.set_title('Input Image')

# 显示预测结果
ax2.bar(texts, probs[0])
ax2.set_ylabel('Probability')
ax2.set_title('CLIP Predictions')

# 调整布局并显示
plt.tight_layout()
plt.show()

# 打印结果
for i, label in enumerate(texts):
    print(f"{label}: {probs[0][i]:.2%}")
