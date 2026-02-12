import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix()) # check these folders
from peepholelib.plots.AllSeeing import AttackExampleVisualizer

BASE = Path.cwd() / "../../../../data/vgg/datasets"

viz = AttackExampleVisualizer(
    base_path=BASE,
    dataset="CIFAR100",
)

saved = viz.save_examples(
    split="test",
    attack="BIMf",
    out_dir=Path.cwd() / "attack_examples",
    n=5,
    mode="successful",
)

print(saved)