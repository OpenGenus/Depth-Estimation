# Depth-Estimation
Depth Estimation - overview

This is a python code demonstrating a sample working of depth estimation using **MiDaS** model.
The MiDaS models are trained on a mix of several depth estimation datasets, including MegaDepth, ReDWeb, and WSVD. These datasets provide diverse and comprehensive training examples, enabling the model to generalize well to a variety of real-world scenarios.

#### Step 1: Install required libraries
```
pip install torch torchvision opencv-python
pip install timm
```
* **Torch** and **torchvision**: These are essential libraries for building and running deep learning models.
* **OpenCV**: A powerful library for image processing tasks.
* **Timm**: A library for PyTorch image models, which provides pre-trained models and transformations.

After installation is complete, write the following code. Make sure to have a sample image to test the code out on ready. Here it is named "example.jpg", change the name of the image to the name assigned to the image you have saved.

#### Step 2: Import libraries
```
import torch
import cv2
import matplotlib.pyplot as plt
import timm
```
* **Matplotlib**: Used for displaying the input image and the resulting depth map.

#### Step 3: Load the MiDaS model
The MiDaS model is used for monocular depth estimation, which predicts depth from a single image.
```
model_type = "DPT_Large"  # MiDaS v3 - Large model
midas = torch.hub.load("intel-isl/MiDaS", model_type)
```
* **MiDaS** stands for **Monocular Depth Estimation for Autonomous Systems**, it provides robust depth predictions from images. The "DPT_Large" model is a variant of MiDaS that uses a large Transformer architecture for enhanced accuracy. MiDaS was developed by Intel.

#### Step 4: Load the transformation pipeline
The transformation pipeline processes the input image to be compatible with the model.
```
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
```
* **Transforms**: These pre-process the input image by resizing, normalizing, and converting it into a tensor. This step is essential to ensure the image format matches what the model expects.

#### Step 5: Load an example image
Load any image you want to estimate the depth for.
```
img_path = "example.jpg"
img = cv2.imread(img_path)
```
* **OpenCV** is used to read the image from the specified file path. Make sure to replace "example.jpg" with the path to your image file.

#### Step 6: Check if the image was loaded successfully
```
if img is None:
    raise ValueError(f"Image at path '{img_path}' could not be loaded. Please check the file path and try again.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```
If the image cannot be loaded, an error message is displayed.

#### Step 7: Apply the transformation to the image
Transform the image to prepare it for input to the model.
```
input_batch = transform(img).unsqueeze(0)
```
* The **transform** function converts the image into a tensor and adds a batch dimension, which is required by the model.

#### Step 8: Ensure the input tensor has the correct shape
```
if len(input_batch.shape) == 5:
    input_batch = input_batch.squeeze(0)  # Remove the extra dimension
```
This step ensures the input tensor has the correct shape, typically (batch_size, channels, height, width).

#### Step 9: Move the input to the GPU if available
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
input_batch = input_batch.to(device)
```
The code checks for GPU availability and moves both the model and input tensor to the GPU for faster processing.

#### Step 10: Perform depth estimation

```
with torch.no_grad():
    prediction = midas(input_batch)
```
The model generates a depth map prediction for the input image without updating model weights (torch.no_grad() ensures no gradients are calculated, reducing memory usage).
 
#### Step 11: Remove the extra dimension and convert to numpy
Convert the prediction to a more usable format.
```
prediction = torch.nn.functional.interpolate(
    prediction.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()
depth_map = prediction.cpu().numpy()
```
* **interpolate** Resizes the depth prediction to match the input image size using bicubic interpolation, which helps maintain detail.
* ```depth_map = prediction.cpu().numpy()``` the prediction is converted from a tensor to a NumPy array for easy manipulation and display.
 
#### Step 12: Display the depth map
Visualize the original image alongside the estimated depth map.
```
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
```
* **Matplotlib** is used to display the original image and the depth map. The depth map is shown using the 'inferno' colormap, which provides a visually appealing representation of depth.
