import torch
import matplotlib.pyplot as plt

out_base = "/srv/newpenny/XAI/generated_data/Kami_attacks/CIFAR100_WRN28-robust/protoscores"

clean   = torch.load(f"{out_base}/protoscores.CIFAR100-test-WRN28-robust.pt")
apgd_ce = torch.load(f"{out_base}/protoscores.CIFAR100-test-APGD-ce-WRN28-robust.pt")
apgd_t  = torch.load(f"{out_base}/protoscores.CIFAR100-test-APGD-t-WRN28-robust.pt")

# just check one layer first
layer = 'relu'

c  = clean[layer].numpy()
ce = apgd_ce[layer].numpy()
t  = apgd_t[layer].numpy()

print(f"Clean   | mean: {c.mean():.3f} | std: {c.std():.3f}")
print(f"APGD-ce | mean: {ce.mean():.3f} | std: {ce.std():.3f}")
print(f"APGD-t  | mean: {t.mean():.3f} | std: {t.std():.3f}")

for layer in list(clean.keys()):
    c  = clean[layer].numpy()
    ce = apgd_ce[layer].numpy()
    t  = apgd_t[layer].numpy()
    print(f"{layer:30s} | clean: {c.mean():.3f} | apgd-ce: {ce.mean():.3f} | apgd-t: {t.mean():.3f}")