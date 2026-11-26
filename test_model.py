import torch
import cv2
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384]
    }
}

print("Loading Depth Anything V2 VITS...")
print("Device:", DEVICE)

model = DepthAnythingV2(**model_configs["vits"])

model.load_state_dict(
    torch.load(
        "checkpoints/depth_anything_v2_vits.pth",
        map_location="cpu"
    )
)

model = model.to(DEVICE).eval()

print("Model loaded successfully!")

# Test with one image
image = cv2.imread("test.jpg")

if image is None:
    print("ERROR: test.jpg not found.")
    print("Put any JPG image in this folder and name it test.jpg")
    exit()

depth = model.infer_image(image)

print("Depth inference successful!")
print("Depth shape:", depth.shape)
print("Depth min:", depth.min())
print("Depth max:", depth.max())