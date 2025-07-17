# First things first, Conda.

We have a share conda, and use a common evironment. 
Add it up to your user's bashrc

```sh
/srv/newpenny/conda/bin/conda init
```

After that, log out and in again and you should be able to use the environment with all necessary packages installed.

```sh
conda activate xai-venv
```

# Download the repositories

I reccom keeping the same dir structure as us (will make it simpler).

Note that there are two repositories, one is your local copy of the `peepholelib`, with the library. The second `XAI` has scripts for making the experiments. 

```sh
cd
mkdir repos
cd repos
git clone git@github.com:SSIGPRO/peepholelib.git
git clone git@github.com:SSIGPRO/XAI.git
```

Go to `XAI`, for making your experiments.

```sh
cd XAI
git checkout rovattinagi
cd src/Rovatti
```
# Reference scripts for your enjoyment

Two important objects are the  `corevectors` and `peepholes`, they contain the datasets with all information you will need. The `corevectors` has an internal structure with the dataset images, model output, labels and etc.

- check `xp_rovatti.py` for an example of how to load the data, with the paths for the already tuned `peepholes`.
- for reference, you can use the script `peepholelib/peepholelib/utils/fine_tune.py` for the training loop.
