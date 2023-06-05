# Continuous diffusion model for Point Cloud Generation

## Requirements

From NERSC, you already have access to almost all libraries from the ```tensorflow module```, to install the remaining libraries you can either [set up your own virtual environment](https://docs.nersc.gov/development/languages/python/nersc-python/) using the tensorflow module as the base or instead using [shifter+docker](https://docs.nersc.gov/development/shifter/how-to-use/). For the latter, see the instructions below.

Pull the image:

```bash
shifterimg -v pull vmikuni/tensorflow:ngc-22.08-tf2-v0
```

Start an interactive job with the image you just loaded:

```bash
salloc -C gpu -q interactive  -t 10 -n 4 --ntasks-per-node=4 --gpus-per-task=1  -A das_repo --gpu-bind=none  --image=vmikuni/tensorflow:ngc-22.08-tf2-v0 --module=gpu,nccl-2.15 
```

After your interactive session starts, let's run a quick training to verify that the current environment is working

```bash
srun --mpi=pmi2 shifter python train.py --data_path /path/to/dowloaded/files
```

If everything works properly, you will see the training starting.

# Evaluating the results

We want to compare different ODE solvers in terms of speed and accuracy. The plotting script can be called as:

```bash
python plot_jet.py --sample --data_path /path/to/dowloaded/files
```

Notice that now we need to have ```pytorch``` installed to load the evaluation metrics from [JetNet](https://github.com/jet-net/JetNet), that you also need to clone.

In case you set up your own virtual environment, add the pytorch library to it as well, otherwise you can install pytorch on your ```PYTHONUSERBASE``` with:

```bash
shifter --image=vmikuni/tensorflow:ngc-22.08-tf2-v0 --module=gpu,nccl-2.15 --env PYTHONUSERBASE=$HOME/.local/perlmutter/my_tf-22.08-tf2-py3_env
pip install --user torch
```

In case you already generated the datasets and just want to redo the metrics, run:

```
python plot_jet.py
```