pip install torch torchvision opencv-python
pip install timm

import torch
import cv2
import matplotlib.pyplot as plt
import timm

model_type = "DPT_Large"  # MiDaS v3 - Large model
midas = torch.hub.load("intel-isl/MiDaS", model_type)

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

img_path = "example.jpg"  # Put name of the image you want to use here
img = cv2.imread(img_path)

if img is None:
    raise ValueError(f"Image at path '{img_path}' could not be loaded. Please check the file path and try again.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).unsqueeze(0)

if len(input_batch.shape) == 5:
    input_batch = input_batch.squeeze(0)  # Remove the extra dimension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
input_batch = input_batch.to(device)

with torch.no_grad():
    prediction = midas(input_batch)

prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()
depth_map = prediction.cpu().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Depth Map")
plt.imshow(depth_map, cmap="inferno")
plt.axis("off")

plt.show()
