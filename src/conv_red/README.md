# Convolutional Layer Dimensionality Reductions

Source code for the paper "A Convolutional Layer Activation Dimensionality Reduction for Out-of-Distribution and Adversarial Attack Detection Methods"

## General syntax and structure

Folders and files containing experiments and configurations:
- `configs`: general configurations (paths, parameters, and auxiliary functions) for models, dimensionality reductions, and analysis.
- `fine_tune_models`: experiment for finetuning models.
- `datasets`: experiment for parsing dataset and applying adversarial attacks.
- `corevectors`: experiment for computing the corevectors.
- `tuning`: experiment for tuning the OoD and AA detection methods.
- `peepholes`: experiment for evaluating the best configuration from tuning.
- `eval`: making plots and computing AUCs.
- `temp_plots`: results are saved here.
- `utils`: auxiliary implementations.
- `Makefile`: shortcuts for running all tests.

The general syntax for the experiments follows:
```sh
python <folder>/<experiment name>.py -m <Model> -r <reduction> -a <analysis> -d <directory>
```
Where:
- `<Model> = [VGG|MobileNet|ResNet|ConvNeXt]` is the model to evaluate.
- `<reduction> = [kernel|toeplitz|avgpooling]` is the dimensionality reduction to use.
- `<analysis> = [MACS|DMD]` is the OoD and AA dettection method.
- `<directory>` is a path in your system to save the corevectors, peepholes, and other generated data.

## Using the Makefile

To run all combinations of models, reduction and analysis, one can use:
```sh
make xp_datasets
make xp_corevectors
make xp_tuning
make xp_peepholes
```

## Finetuning models

To finetune the models to `CIFAR100` use:
```sh
python fine_tune_models/xp_finetune_model.py -m <model>
```

## Reproducing plots

To plot the `AUC`s scatter plot from the paper run:
```sh
python eval/aucs.py -d <directory>
```

