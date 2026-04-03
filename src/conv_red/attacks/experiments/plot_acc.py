import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix()) # check these folders
from peepholelib.plots.attack_compare import ThreatModelComparator

BASE = Path.cwd()/"../../../data/robustwrn/datasets"

threat_models = {
    # "Linf": ["APGDf", "BIMf", "PGDf"],
    "Linf": ["auto"],
}

cmp = ThreatModelComparator(BASE, dataset="CIFAR100", threat_models=threat_models)

results = cmp.plot(split="test", out_dir=Path.cwd() / "attack_compare_plots")
print(results)