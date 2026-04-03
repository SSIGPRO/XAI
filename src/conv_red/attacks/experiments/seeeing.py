import sys
from pathlib import Path
sys.path.insert(0, (Path.home() / 'repos/peepholelib').as_posix())

from peepholelib.plots.AllSeeing import AttackExampleVisualizer

BASE = Path.cwd() / "../../../../data/wrn/datasets"

viz = AttackExampleVisualizer(
    base_path=BASE,
    dataset="CIFAR100",
)

saved = viz.save_examples(
    split="test",
    attack="APGDf",
    out_dir=Path.cwd() / "attack_examples",
    indices=2, # indices=[17, 25, 44],
)

print(saved)