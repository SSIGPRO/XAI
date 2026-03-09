import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix()) # check these folders
from peepholelib.plots.attack_compare import ThreatModelComparator

BASE = Path.cwd()/"../../../data/vgg/datasets"

threat_models = {
    # "Linf": ["BIM", "PGD", "APGD"],
    # "L2":   ["CW", "DF", "APGD2"],
    # "Linf": ["APGDl","APGDr", "APGDu", "APGDf", "BIMl", "BIMr","BIMf", "PGDl", "PGDr", "PGDf"],
    # "L2": ["APGD2l","APGD2r", "APGD2u", "APGD2f", "DF"],
    # "Linf": ["PGDl","PGDr","PGDf"],
    "Linf": ["APGDf", "BIMf", "PGDf"],
}

cmp = ThreatModelComparator(BASE, dataset="CIFAR100", threat_models=threat_models)

results = cmp.plot(split="test", out_dir=Path.cwd() / "attack_compare_plots")
print(results)