import torch
import matplotlib.pyplot as plt

path = "/home/arshakumari/repos/XAI/src/Conv_NeXt/../../data/checkpoints/20260331_170847/best_model/best_model_config.pt"

data = torch.load(path, map_location="cpu")

train_losses = data["train_losses"]
val_losses = data["val_losses"]

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")

plt.legend()
plt.grid()

plt.savefig("loss_curve.png")
print("Saved as loss_curve.png")