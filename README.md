# Ultrametric-benchmark-v2

This is the repo for the Ultrametric Benchmark (UB). This is a benchmark for Continual Learning (CL) that quantifies the phenomenon of Catastrophic Forgetting (CF).
The main idea is to present samples to a network ordered in a ultrametric (UM) sequence that approximates class distribution in real-world events. 
The sequences can then be shuffled destroying specific correlation timescales to quantify each timescale's contribution to CF.

## Project folder layout

```
home  
    ├── env  
    │   ├── env1_no_builds.yml  
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
    │   │       ├── datasets.py             #generate dataset
    │   │       ├── functions_markov.py     #functions to generate the markov chain
    │   │       ├── network.py              #setup the network and train
    |   |       ├── ultrametric_callback.py #callbacks for the network
    │   │       ├── UltrametricTree.py  
    │   │       └── util_functions.py  
    │   ├── data    
    │   │   ├── MNIST  
    │   │   └── saved_chains  
    │   ├── notebooks  
    │   │   ├── postprocessing_metrics.ipynb  
    │   │   ├── ultrametric_chain.ipynb  
    ├── scratch  
    │   ├── logs  
  ```

`scripts` is where the `.run` files for launching jobs on the cluster are located.  

## Install

In order to install the repo please clone it to your local machine. Then upload it to your GPU cluster (the code will also run on CPU albeit very slowly). Upload using the command: `scp -r $LOCAL_PATH/repo user@hostname:$REMOTE_PATH/`. Then you will have to install Ananconda on your remote cluster. Then install the included conda environment in order to have access to the necessary packages. This is done with `conda env create -f home/env/env1_no_builds.yml`. 
For the datasets, you will have to upload the MNIST dataset to the dedicated MNIST folder. For the artificial dataset you will have to generate the corresponding UM chains in `ultrametric_chain.ipynb` in the chapter "generate sequences to save". Please upload these chains to the cluster as well, under `home/data/saved_chains`. The code will also create a chain on the spot if none is provided but it is better to work with pre-computed chains for efficiency.
The install is now complete

## Run

The run command launches an array of SLURM jobs. THis is useful for parallelisation. The commands is structured as follows:  
`sbatch launch_arr.run -e 200 -f 1.9 -m "split" -s 400 -l true "h" "b_len" "d" "lr" "seed"`
 where after the named command line options in order:  
 ```
 1) `h`: int, is the number of hidden neurons in the hidden layer.  
 2) `b_len`: int, is the length of the shuffling blocks.
 3) `d`: int, is the tree depth
 4) `lr`: float, is the learning rate
 5) `seed`: int, is the seed number for the repetitions.
```
Note that all of these fields are lists so you can easily launch multiple runs with different parameter values in a single command. The `launch_arr.run` file then calls the file `um_arr.run` for each individual run.

And the named command line options are as follows:  
```
e) eval_freq : int, frequency at which we evaluate the model and shuffle the sequence.  
f) ef_factor : float, exponential growth factor for eval_freq.  
m) mode : string, one of "um" (ultrametric), "split" or "rand" for the completely ranodmised case.  
s) s_len : int (optional), for the split case it is the length of the splits.  
c) ckpt_path : string (optional), in the case of the restart of a previous interrupted run, it is the checkpoint path from which we want to restart.
k) keep_correlations : bool (optional), whether we want to keep the spatial correlations in the dataset, default=False.  
l) stoch_s_len : bool (optional), whether to include some stochasticity in the s_len parameter, default=False.  
```

## Datasets

In its current state the code can either be run on an artificial dataset we create by random walk on an ultrametric tree. We also offer the possibility to run it on MNIST.
To launch it on the artificial dataset use `launch_arr.run` in the command and `launch_mnist.run` for MNIST.

## Relaunching jobs

If your cluster has a time limit or if your jobs were interrupted, we created an automated way to find jobs that have not converged 
and create the appropriate relaunch commands to resume these runs. You have to go to the file `home/repo/code/scripts/launch_collect.run`.
You have to put in the intervall in job_ids that you would like to consider for relaunch, and then do `sbatch launch_collect.run`.
This will create a file named `relaunch_commands.txt` which contains the commands to be relaunched. To relaunch them, simply do `sbatch launch_sbatch.run`.

## Logs

The logging system is structured as follows:
```
├── logs  
│   ├──3129405_1108_1424_synth_split_b1_d5_stochs400_h60_lr1.0_rep9
        │   │   └── metrics
        │   │       └── fold_9_part0
        │   │           ├── events.out.tfevents.1660255228.g043.264498.1
        │   │           └── hparams.yaml
```
In the log folder name, we have `{job_id}_{time}_{date}_{ds_name}_{mode}_b{b_len}_d{depth}_{stochasticity}{s_len}_{h}_{lr}_rep{seed}`.
Then we have the folder `fold_{seed}_part_{relaunch_nb}`. Then the `.1` file is the actual logs, and `hparams.yaml` contains the hyperparameters used 
for this run.

We recommend downloading the logs from the cluster using `scp -r user@hostname:home/scratch/logs $LOCAL_PATH/logs`.

## Plots

All the plots in the paper are generated in `ultrametric_chain.ipynb` for the auto-correlation plots (function `plot_autocorr`), and in `postprocessing_metrics.ipynb` for the 
validation accuracy curves (function `plot_runs_w_regex`) and derived metrics (function `summary_plot`).



