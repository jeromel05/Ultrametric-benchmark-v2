import argparse
import os, sys
from os.path import join
import csv
import yaml
import re

import numpy as np
import pandas as pd
curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, join(curr_path,"../utils/"))

from datetime import datetime
import time

empty_hparams = {   'b_len': 0, 'batch_size': 2, 'eval_freq': 1000, 'eval_freq_factor': 1.7,
                    'hidden_size': 60, 'input_size': 200, 'last_val_step': 0, 'lr': 2.0,
                    'lr_scheduler': 'null', 'mode': 'um', 'nb_classes': 32, 'no_reshuffle': False,
                    'optimizer': 'sgd', 's_len': 0, 'val_step': 100 }

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logfolder', type=str, default='../../../scratch/logs/', help="Folder to save the data")
    parser.add_argument('--outfolder', type=str, default='../../../scratch/', help="Folder to save the .out files")
    parser.add_argument('--job_info_csv', type=str, default='../../../scratch/job_info.csv', help="Folder to save the .out files")
    parser.add_argument('-j','--job_range', nargs='+', help='range of job_ids to analyse', required=True)

    args = parser.parse_args()
    print("All args: ", args)
    assert(len(args.job_range) % 2 == 0) # enforce parity
    job_range = args.job_range
    logfolder = args.logfolder
    outfolder = args.outfolder
    job_range = [int(el) for el in job_range]
    list_jobs_to_analyse = [el for i in range(len(job_range)-1) for el in np.arange(job_range[i], job_range[i+1], step=1)]

    all_logdirs = sorted(os.listdir(logfolder))
    all_outfiles = sorted(os.listdir(outfolder))
    log_out_files_dict = construct_log_out_files_dict(all_logdirs, all_outfiles, list_jobs_to_analyse)

    if(os.path.isfile(args.job_info_csv)):
        job_info_df = pd.read_csv(args.job_info_csv)
    else:
        job_info_df = pd.DataFrame()

    results_dict = dict() #pd.DataFrame(index=['job_id'], columns=[['best_acc', 'best_loss', 'best_step', 'log_path', 'hash']])
    failed_jobs_list = []

    for job_id, (outfile, logpath) in log_out_files_dict.items():
        outfile =  join(outfolder, outfile)
        log_path =  join(logfolder, logpath)
        
        if not os.path.isdir(log_path) or not os.path.isfile(outfile):
            print(f'ERROR: job {job_id} logs not found')
            failed_jobs_list.append((job_id, ''))
            print(f'Missing log_path: {log_path}, outfile : {outfile}')
            break

        
        metric_dict = dict()
        with open(outfile, 'r') as csvfile:
            csvreader = reversed(list(csv.reader(csvfile, delimiter=',')))
            for i, row in enumerate(csvreader):
                if i < 100:
                    # 'Val[ 0-9a-z_.]*step: ([0-9]+),[ 0-9a-z_.]*loss: ([0-9.]+)'
                    matched = re.search('acc: ([0-9.]+)', row[-1])
                    if matched:
                        for metric in row:
                            matched_key_value = re.search('([a-z]+): ([0-9.]+)', metric)
                            key1 = matched_key_value.group(1)
                            value1 = float(matched_key_value.group(2))
                            metric_dict[key1] = value1
                        break
                        
            if len(metric_dict) == 0:
                print(f'ERROR: job {job_id} has failed to run')
                failed_jobs_list.append((job_id, log_path))

            hash1 = get_hash(log_path)
            results_dict[job_id] = [metric_dict, log_path, hash1]
        
    print(f'Out of {len(list_jobs_to_analyse)} logs to analyse, {len(results_dict)} logs were collected , {len(failed_jobs_list)} failed completely')
    if len(results_dict) == 0:
        print('No logs found to analyse: exiting')
        sys.exit()

    results_df = pd.DataFrame({'metrics': [el[0] for el in results_dict.values()], 'log_path': [el[1] for el in results_dict.values()],
                                'hash': [el[2] for el in results_dict.values()]}, index=[el for el in results_dict.keys()])

    # Merges loaded job info df and results df, 
    pd.join()

    results_df = pd.concat([results_df.drop(['metrics'], axis=1), results_df['metrics'].apply(pd.Series)], axis=1).copy()
    if 'epoch' in results_df.columns:
        results_df['epoch'] = results_df['epoch'].astype(int)
    if 'step' in results_df.columns:
        results_df['step'] = results_df['step'].astype(int)

    jobs_to_relaunch_df = results_df.loc[results_df['acc'] < 0.95].copy()
    print(f'Out of {results_df.shape[0]} logged jobs, {jobs_to_relaunch_df.shape[0]} have not converged')

    curr_running_job_ids = []
    with open('running_jobs.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            job_id = int(row[1])
            if job_id in jobs_to_relaunch_df.index:
                curr_running_job_ids.append(job_id)
    

    jobs_to_relaunch_df.loc[:, 'curr_running'] = np.zeros(jobs_to_relaunch_df.shape[0])
    jobs_to_relaunch_df.loc[curr_running_job_ids,'curr_running'] = 1
    jobs_to_relaunch_df.loc[:, 'relaunched_counter'] = np.zeros(jobs_to_relaunch_df.shape[0])
    jobs_to_relaunch_df.loc[jobs_to_relaunch_df['step'] > 110000, 'relaunched_counter']= 1

    jobs_to_relaunch_df = jobs_to_relaunch_df.loc[(jobs_to_relaunch_df['curr_running'] == 0) & (jobs_to_relaunch_df['relaunched_counter'] == 0)]
    jobs_to_relaunch_df['hparams'] = jobs_to_relaunch_df['log_path'].apply(get_hparams_from_file)
    jobs_to_relaunch_df = pd.concat([jobs_to_relaunch_df.drop(['hparams'], axis=1), jobs_to_relaunch_df['hparams'].apply(pd.Series)], axis=1)
    print(jobs_to_relaunch_df.head(2))
    print(jobs_to_relaunch_df.columns)

    sbatch_relaunch_commands = []
    for index, row in jobs_to_relaunch_df.iterrows():
        print(row)
        ckpt_path = '/burg/home/jl6181/home/scratch/logs/2626672_2106_1905_synth_um_b0_d5_h60_lr2.0_rep17/fold_17/step\=2709_val_acc\=0.7969.ckpt'
        eval_freq = row['eval_freq']
        eval_freq_factor = row['eval_freq_factor']
        mode = row['mode']
        hidden_size = row['hidden_size']
        b_len = row['b_len']
        lr = row['lr']
        hidden_size = row['hidden_size']
        depth = row['depth']
        s_len = row['s_len']
        keep_correlations = row['keep_correlations']
        stoch_s_len = row['stoch_s_len']
        rep_nb = int(re.search('rep([0-9]+)', ckpt_path).group(1))

        sbatch_command = f'sbatch launch_arr.run -e {eval_freq} -f {eval_freq_factor} -m {mode} -c {ckpt_path} -s {s_len} -k {keep_correlations} -l {stoch_s_len} "{hidden_size}" "{b_len}" "{depth}" "{lr}" "{rep_nb}"'
        #sbatch launch_arr.run -e 1000 -f 1.7 -m "um" -c "/burg/home/jl6181/home/scratch/logs/2626672_2106_1905_synth_um_b0_d5_h60_lr2.0_rep17/fold_17/step\=2709_val_acc\=0.7969.ckpt" "60" "0" "5" "2.0" "17"
        sbatch_relaunch_commands.append(sbatch_command)

    print('DONE')
    print(sbatch_relaunch_commands)
    return sbatch_relaunch_commands
        

def get_hash(ckpt_name):
    if '/' in ckpt_name:
        ckpt_name = ckpt_name[ckpt_name.rfind('/'):]
    patt = "[./]*(?:merged_)*([0-9]+)_([0-9]+)_[0-9]+_[a-z]+_((?:um|rand|split)_b[0-9]+_*(d[0-9]+)*_*(s[0-9]*)*_+h[0-9]+_lr[0-9.]+_rep[0-9]+)"
    matched = re.match(patt, ckpt_name)
    if matched:
        hash1 = matched.group(3) # create hash
    else:
        print(f'Not able to hash {ckpt_name}')
        hash1=''
    return hash1

def construct_log_out_files_dict(all_logdirs, all_outfiles, list_jobs_to_analyse):
    log_out_files_dict = dict()
    for logdir in all_logdirs:
        matched = re.search('(2[0-9]{6})_([0-9]{4}_){2}', logdir)
        if matched:
            job_id_str = matched.group(1)
            if int(job_id_str) in list_jobs_to_analyse:
                for outfile in all_outfiles:
                    if job_id_str in outfile:
                        log_out_files_dict[int(job_id_str)] = (outfile, logdir) 
    return log_out_files_dict

def get_hparams_from_file(log_path):
    rep_nb = int(re.search('rep([0-9]+)', log_path).group(1))
    hparams_path = join(log_path, 'metrics', f'fold_{rep_nb}', 'hparams.yaml')
    hparams_dict=empty_hparams
    
    if os.path.isfile(hparams_path):
        with open(hparams_path, "r") as stream:
            try:
                hparams_dict=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    else:
        print(f'hparams.yaml file not found {hparams_path}')
    print('hparams_dict', hparams_dict)
    return hparams_dict

if __name__ == '__main__':
    run()
