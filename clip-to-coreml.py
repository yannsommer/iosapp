import torch
import clip
from PIL import Image
import coremltools as ct

# 加载CLIP模型
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# 准备示例输入
example_image_path = "IMG_4336.jpg"
example_image = preprocess(Image.open(example_image_path)).unsqueeze(0).to(device)
example_texts = ["a dog", "a cat"]
example_text = clip.tokenize(example_texts).to(device)

# 追踪模型
traced_model = torch.jit.trace(model, (example_image, example_text))

# 转换为Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.ImageType(name="image", shape=example_image.shape),
        ct.TensorType(name="text", shape=example_text.shape)
    ],
    outputs=[
        ct.TensorType(name="image_features"),
        ct.TensorType(name="text_features")
    ]
)

# 保存模型
mlmodel.save("CLIP.mlmodel")
