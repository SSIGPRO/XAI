import torch
import os

bin_path = "/home/claranunesbarrancos/repos/XAI/src/vggBN/pytorch_model_vgg.bin"

checkpoint = torch.load(bin_path, map_location="cpu")
print("Checkpoint type:", type(checkpoint))
if isinstance(checkpoint, dict):
    print("Checkpoint keys:", checkpoint.keys())

save_path = "/srv/newpenny/XAI/models/vgg16_BN_dataset=CIFAR100.pth_optim=SGD_scheduler=CosineAnnealingLR.pth"

# If it's a plain state_dict, save it as-is
# If it's a full checkpoint, extract the model weights
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    torch.save(checkpoint["model_state_dict"], save_path)
else:
    torch.save(checkpoint, save_path)

print(f"Model saved to: {save_path}")
