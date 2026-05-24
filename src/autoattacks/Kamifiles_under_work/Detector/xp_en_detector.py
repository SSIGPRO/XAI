import sys
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from peepholelib.utils.Energy_Utils.Ensemble_detector import (
    load_td,
    compute_tail_energy,
    calibrate_threshold,
    evaluate_ensemble,
    plot_threshold,
    print_calibration_info,
    print_results_table,
)


# Configs

DATASET        = "CIFAR100"
SPLIT          = "test"
MODEL_STANDARD = "WRN28-standard"
MODEL_ROBUST   = "WRN28-robust"
ATTACKS        = ["APGD-ce", "APGD-t"]

LAYER = "block1.layer.2.conv2"
TAIL  = 100

#   "fpr"       : fix clean false-positive rate
#   "asr"       : fix ASR 
#   "clean_acc" : fix ensemble clean accuracy

CALIBRATION_MODE   = "fpr"

TARGET_FPR         = 0.0478          # used when mode == "fpr"
TARGET_ASR         = 0.4656          # used when mode == "asr"
CALIBRATION_ATTACK = "APGD-t"        # used when mode == "asr"
TARGET_CLEAN_ACC   = 0.7257          # used when mode == "clean_acc"

# paths (Adjust if in another folder, right now points to Kami_attacks)

_ROOT   = Path("/srv/newpenny/XAI/generated_data/Kami_attacks")
DS_BASE = _ROOT / "datasets" / DATASET
CV_BASE = _ROOT / f"{DATASET}_{MODEL_STANDARD}" / "corevectors"
PLOT_DIR = _ROOT / f"{DATASET}_{MODEL_STANDARD}" / "tail_energy_ensemble"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


#path helpers

def _ds_path(model, attack=None):
    name = (f"dss.{DATASET}-{SPLIT}.{attack}-{model}"
            if attack else f"dss.{DATASET}-{SPLIT}.{model}")
    return DS_BASE / name

def _adv_robust_path(attack):
    return DS_BASE / f"dss.{DATASET}-{SPLIT}.{attack}-{MODEL_STANDARD}.{MODEL_ROBUST}"

def _cv_path(attack=None):
    name = (f"corevectors.svd.{DATASET}-{SPLIT}-{attack}-{MODEL_STANDARD}"
            if attack else f"corevectors.svd.{DATASET}-{SPLIT}-{MODEL_STANDARD}")
    return CV_BASE / name


# load data

print("=" * 70)
print("Loading datasets and corevectors")
print("=" * 70)

clean_ds_std  = load_td(_ds_path(MODEL_STANDARD))
clean_ds_rob  = load_td(_ds_path(MODEL_ROBUST))
clean_ds_base = load_td(DS_BASE / f"dss.{DATASET}-{SPLIT}")
clean_cv      = load_td(_cv_path())

attack_ds_std = {a: load_td(_ds_path(MODEL_STANDARD, a)) for a in ATTACKS}
attack_ds_rob = {a: load_td(_adv_robust_path(a))          for a in ATTACKS}
attack_cv     = {a: load_td(_cv_path(a))                  for a in ATTACKS}

print("All files loaded.\n")


# clean baseline

clean_correct_std_all = clean_ds_std["result"][:].bool().cpu().numpy()
clean_correct_rob_all = clean_ds_rob["result"][:].bool().cpu().numpy()
clean_labels          = clean_ds_base["label"][:].cpu().numpy()

good_mask          = clean_correct_std_all
good_indices       = np.where(good_mask)[0]
base_clean_acc_std = good_mask.mean()
base_clean_acc_rob = clean_correct_rob_all.mean()

print(f"Standard model clean accuracy : {base_clean_acc_std:.4f}  ({good_mask.sum()}/{len(good_mask)})")
print(f"Robust   model clean accuracy : {base_clean_acc_rob:.4f}\n")


# tail energy

mask_tensor = torch.from_numpy(good_mask)

clean_energy = compute_tail_energy(clean_cv, LAYER, mask_tensor, TAIL)

attack_energy           = {}
attack_correct_std      = {}
attack_correct_rob      = {}

for attack in ATTACKS:
    attack_energy[attack] = compute_tail_energy(
        attack_cv[attack], LAYER, mask_tensor, TAIL,
    )
    attack_correct_std[attack] = (
        attack_ds_std[attack]["result"][:].bool().cpu().numpy()[good_indices]
    )
    robust_pred = attack_ds_rob[attack]["pred"][:].cpu().numpy()
    attack_correct_rob[attack] = (robust_pred == clean_labels)[good_indices]


# threshold

print("=" * 70)
print(f"Calibrating threshold  (mode = '{CALIBRATION_MODE}')")
print("=" * 70)

clean_correct_rob_masked = clean_correct_rob_all[good_indices]

calib_kwargs = dict(
    clean_energy         = clean_energy,
    clean_correct_robust = clean_correct_rob_masked,
    base_clean_acc       = base_clean_acc_std,
)

if CALIBRATION_MODE == "fpr":
    target = TARGET_FPR
    calib_kwargs["target_fpr"] = TARGET_FPR

elif CALIBRATION_MODE == "asr":
    target = TARGET_ASR
    calib_kwargs.update(
        target_asr                  = TARGET_ASR,
        calib_adv_energy            = attack_energy[CALIBRATION_ATTACK],
        calib_adv_correct_standard  = attack_correct_std[CALIBRATION_ATTACK],
        calib_adv_correct_robust    = attack_correct_rob[CALIBRATION_ATTACK],
    )

elif CALIBRATION_MODE == "clean_acc":
    target = TARGET_CLEAN_ACC
    calib_kwargs["target_clean_acc"] = TARGET_CLEAN_ACC

threshold, calib_info = calibrate_threshold(CALIBRATION_MODE, **calib_kwargs)
print_calibration_info(calib_info, CALIBRATION_MODE, target)


# plot threshold

safe_layer = LAYER.replace(".", "_")
plot_name  = f"ensemble_{MODEL_STANDARD}_{safe_layer}_{CALIBRATION_MODE}.png"
plot_path  = PLOT_DIR / plot_name

plot_threshold(
    clean_energy    = clean_energy,
    attack_energies = attack_energy,
    threshold       = threshold,
    empirical_fpr   = calib_info["empirical_fpr"],
    layer           = LAYER,
    save_path       = plot_path,
    title_extra     = (
        f"mode={CALIBRATION_MODE}  target={target}  "
        f"FPR={calib_info['empirical_fpr']:.2%}"
    ),
)
print(f"\nPlot saved: {plot_path}")


# evaluate

print("\n" + "=" * 70)
print("Results")
print("=" * 70)

results = [
    evaluate_ensemble(
        attack_name          = attack,
        clean_energy         = clean_energy,
        adv_energy           = attack_energy[attack],
        adv_correct_standard = attack_correct_std[attack],
        adv_correct_robust   = attack_correct_rob[attack],
        threshold            = threshold,
    )
    for attack in ATTACKS
]

print_results_table(results, calib_info)