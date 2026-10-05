from gaussianSplat import GaussianSplat2D as gs2D
from gaussianSplat import GaussianSplat3D as gs3D
from PIL import Image
import json
import torch
import numpy as np


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()

# images = ["cat", "astronaut", "coffee"]
# Ns = [256, 1024, 4096]
# for image in images:    
#     img = Image.open(f"./Gaussian Splatting/data/Images/{image}.png")
#     for N in Ns:
#         final = gs2D.optimize2D(img, N, N)
#         Image.fromarray(final).save(f"./Gaussian Splatting/data/Images/Results/gs_{N}_{N}_{image}.png")

# print("densify test:")
# N = 1024
# img = Image.open(f"./Gaussian Splatting/data/Images/cat.png")
# final = gs2D.optimize2D(img, 256, 1024)
# Image.fromarray(final).save(f"./Gaussian Splatting/data/Images/Results/gs_256_1024_cat.png")

class Camera:
    def __init__(self, image, R, t, K, H, W, file, xy):
        self.image = image
        self.R = R
        self.t = t
        self.K = K
        self.H = H
        self.W = W
        self.file = file
        self.xy = xy


with open("./Gaussian Splatting/data/spheres/cameras.json", "r") as f:
    data = json.load(f)

H = data["height"]
W = data["width"]
K = torch.tensor(data["K"], dtype=torch.float32)

train_cameras = []
val_cameras = []

for frame in data["frames"]:
    name = frame["file"]
    image = torch.from_numpy(
        np.array(Image.open(f"./Gaussian Splatting/data/spheres/{name}").convert("RGB"))
    ).float().to(device) / 255.0

    R = torch.tensor(frame["R_wc"], dtype=torch.float32, device=device)
    t = torch.tensor(frame["t"], dtype=torch.float32, device=device)
    H, W, _ = image.shape
    xy = gs2D.pixel_grid(H, W, device)

    train_cameras.append(Camera(image, R, t, K, H, W, name, xy))

for frame in data["val_frames"]:
    name = frame["file"]
    R = torch.tensor(frame["R_wc"], dtype=torch.float32, device=device)
    t = torch.tensor(frame["t"], dtype=torch.float32, device=device)
    K = torch.tensor(data["K"], dtype=torch.float32, device=device)
    image = torch.from_numpy(
        np.array(Image.open(f"./Gaussian Splatting/data/spheres/{name}").convert("RGB"))
    ).float().to(device) / 255.0
    H, W, _ = image.shape
    xy = gs2D.pixel_grid(H, W, device)

    val_cameras.append(Camera(image, R, t, K, H, W, name, xy))

N = 2048
budget = 4096
final_imgs = gs3D.optimize3D(train_cameras, val_cameras, N, N)

for res in final_imgs:
    image, file = res
    filename = file.split("/")[-1]
    number = filename.split(".")[0]
    viewpoint = f"train_{int(number)}"
    Image.fromarray(image).save(f"./Gaussian Splatting/data/Images/Results/sphere_gs_{N}_{budget}_viewpoint_{viewpoint}.png")