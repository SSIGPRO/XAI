from tensordict import PersistentTensorDict

ori = PersistentTensorDict.from_h5("dss.CIFAR100-test", mode="r")
clean = PersistentTensorDict.from_h5("dss.CIFAR100-test.clean", mode="r")
auto = PersistentTensorDict.from_h5("dss.CIFAR100-test.auto", mode="r")

labels = ori["label"]
pred_clean = clean["pred"]
adv_images = auto["adv_image"]

print(f"Clean accuracy: {(pred_clean == labels).float().mean().item():.4f}")
print(f"Samples: {labels.shape[0]}")

ori.close()
clean.close()
auto.close()