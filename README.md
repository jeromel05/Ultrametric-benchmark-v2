# Ultrametric-benchmark-v2

This is the repo for the Ultrametric Benchmark (UB). This is a benchmark for Continual Learning (CL) that quantifies the phenomenon of Catastrophic Forgetting (CF).
The main idea is to present samples to a network ordered in a ultrametric (UM) sequence that approximates class distribution in real-world events. The sequences can then be shuffled destroying specific correlation timescales to quantify each timescale's contribution to CF.

## Project folder layout

 home
    ├── env
    │   ├── env1_no_builds.yml
    │   └── env1.yml
    ├── repo
    │   ├── code
    │   │   ├── scripts
    │   │   │   ├── launch_arr.run
    │   │   │   ├── launch_collect.run
    │   │   │   ├── launch_mnist.run
    │   │   │   ├── launch_sbatch.run
    │   │   │   ├── other_scripts
    │   │   │   │   ├── launch_rand_arr.run
    │   │   │   │   └── rand_arr.run
    │   │   │   ├── test_scripts
    │   │   │   │   ├── test_arr.run
    │   │   │   │   └── test_launch_arr.run
    │   │   │   ├── um_arr.run
    │   │   │   └── um_mnist.run
    │   │   ├── src
    │   │   │   ├── collect_job_info.py
    │   │   │   ├── running_jobs.csv
    │   │   │   └── run.py
    │   │   └── utils
    │   │       ├── custom_callbacks.py
    │   │       ├── datasets.py
    │   │       ├── Datasets.py
    │   │       ├── functions_markov.py
    │   │       ├── network.py
    |   |       ├── ultrametric_callback.py
    │   │       ├── UltrametricTree.py
    │   │       └── util_functions.py
    │   ├── data
    │   │   ├── MNIST
    │   │   └── saved_chains
    │   ├── scratch
    │   │   ├── logs
    │   ├── notebooks
    │   │   ├── postprocessing_metrics.ipynb
    │   │   ├── ultrametric_chain.ipynb
    

`scripts` is where the `.run` files for launching jobs on the cluster are located.

## Install

In order to install the repo please clone is to your local machine. Then upload it to your GPU cluster (the code will also run on CPU albeit very slowly). Upload using the command: `scp -r $LOCAL_PATH/repo user@hostname:$REMOTE_PATH/`. Then you will have to install Ananconda on your remote cluster. Then install the included conda environment in order to have access to the necessary packages. This is done with `conda env create -f home/env/env1_no_builds.yml`. 
For the datasets, you will have to upload the MNIST dataset to the decicated MNIST folder. For the artificial dataset you will have to generate the corresponding UM chains in `ultrametric_chain.ipynb` in the chapter generate sequences to save. Please upload these chains to the cluster as well, under `home/data/saved_chains`. The code will also create a chain on the sport if none is provided but it is better to work with pre-computed chains for efficiency.
The install is now complete

## Run

The run command launches an array of SLURM jobs. THis is useful for parallelisation. The commands is structured as follows:
`sbatch launch_arr.run -e 200 -f 1.9 -m "split" -s 400 -l true "60" "400 700 1400 2100" "5" "2.0" "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39"`

And the command line options are as follows:
`
e) eval_freq : int, frequency at which we evaluate the model and shuffle the sequence.
f) ef_factor : float, exponential growth factor for eval_freq.
m) mode : string, one of "um" (ultrametric), "split" or "rand" completely ranodmised case.
s) s_len : int (optional), for the split case it is the length of the splits.
c) ckpt_path : string (optional), in the case of the restart of a previous interrupted run, it is the checkpoint path from which we want to restart.
k) keep_correlations : bool (optional), whether we want to keep the spatial correlations in the dataset, default=False.
l) stoch_s_len : bool (optional), whether to include some stochasticity in the s_len parameter, default=False.
`

`


## Datasets

In its current state the code can either be run on an artificial dataset we create by random walk on an ultrametric tree. We also offer the possibility to run it on MNIST.

The 
