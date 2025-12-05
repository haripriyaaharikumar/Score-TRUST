import os
from os.path import join, abspath
import subprocess
from subprocess import Popen

import sys

from utils.general_str import get_arg_string
from global_settings import PYTHON_EXE, RESULTS_DIR, PYTORCH_DATA_DIR

dataset = "cifar10"
dataset_dir = join(PYTORCH_DATA_DIR, dataset)
output_dir = abspath(join(RESULTS_DIR, "pure", "{}".format(dataset), "pytorch"))

"This is to optimize the original dataset to get the delta_x value for any given model"
####
# PROGRAM TO FIND del_X for CIFAR10 DATASET
###
# Default settings
# =============================== #
DEFAULT_CONFIG = {
    # --------------------- #
    # dataset
    # --------------------- #
    "dataset": dataset,
    "data_root": dataset_dir,
    "output_dir": output_dir,
    "workers": 2,
    # --------------------- #
    # architecture
    # --------------------- #
    "model": "SameClassDelta",
    "net": "pre_resnet18",
    "arch": "vinai",
    ###################################
    ### PURE CIFAR10 model####
    ###################################
    # -------------------- #
    ##  Trained Model : Pretrained or trained from scratch
    "model_output_dir":
        "/trained_model/base_model/lr0.001_wdlrschl_epochs250_btchsize128",  
    "model_load_step": 97500, ##Pure
    "model_load_best": False,
    # number of train/test samples
    # --------------------- #
    "train_seed": 1234,
    "target_class": 0, #not relevant
    ##training samples for the attack
    # --------------------- #

    # training
    # --------------------- #
    "epochs": 10000,
    "batch_size": 10,
    # --------------------- #

    # optimizers
    # --------------------- #
    "optim": "adam",
    "lr": 0.001,
    "weight_decay": 0.0,
    "beta1": 0.9,
    "beta2": 0.999,
    "schedule_lr": False,
    # --------------------- #

    # settings
    # --------------------- #
    "norm_type": "l1",
    "use_tanh": True,
    # --------------------- #

    # coefficients
    # --------------------- #
    "xent": 1.0,
    "norm_reg": 0.001,
    "prob_reg":0.0,
    "tv_reg":0.0,
    "T":5.0,
    # --------------------- #
    # freq
    # --------------------- #
    "log_freq": 1000,
    "plot_freq": 1000,
    "save_freq_epoch": 1000,
    "eval_freq_epoch": 1000,
    # --------------------- #
}
# =============================== #
run_config = {
        "run": f"0_class",
        "probability":1.0,
        "model": "SameClassDelta",
        "force_rm_dir": True,
    }
    # =============================== #

config = DEFAULT_CONFIG
config.update(run_config)
arg_str = get_arg_string(config)

print("Running arguments: [{}]".format(arg_str))
run_command = "{} ./train.py {}".format(PYTHON_EXE, arg_str).strip() 
subprocess.call(run_command, shell=True)