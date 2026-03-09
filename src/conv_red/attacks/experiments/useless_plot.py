import sys
from pathlib import Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix()) # check these folders
sys.path.insert(0, (Path.home()/'repos/XAI/src/conv_red').as_posix())
from peepholelib.plots.Atk_plots import AttackDatasetEvaluator 


evaluator = AttackDatasetEvaluator(
    base_path=Path.cwd()/'../../../data'/'vgg'/'datasets',
    dataset_name="CIFAR100",
    split="val"
)

results = evaluator.evaluate([
    "CIFAR100",
    "APGD",
    "BIM",
    "CW"
])

print(results)

evaluator.plot()