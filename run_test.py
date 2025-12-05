import os
from os.path import join, abspath
import subprocess
from utils.general_str import get_arg_string
from global_settings import PYTHON_EXE, RESULTS_DIR, PYTORCH_DATA_DIR

dataset = "cifar10"
dataset_dir = join(PYTORCH_DATA_DIR, dataset)
output_dir = abspath(join(RESULTS_DIR, "pure", "{}".format(dataset), "pytorch"))


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

    # trained x+del_x model
    "trust_model_output_dir":
        "/trained_model/trust_model/10000_lr0.001_norml1reg0.001_Temp5.0",
    "trust_model_load_step": 1000000,
    "trust_model_load_best": False,

    # --------------------- # THE BASE MODEL
    "model_output_dir":
        "/trained_model/base_model/lr0.001_wdlrschl_epochs250_btchsize128", 
    "model_load_step": 97500,
    "model_load_best": False,
 

    # number of train/test samples
    # --------------------- #
    "train_seed": 1234,
    "target_class": 0,
    # --------------------- #
    # training
    # --------------------- #
    "batch_size": 64,
    # --------------------- #

    # optimizers
    # --------------------- #
    "optim": "adam",
    "lr": 0.01,
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
    "norm_reg": 1.0, 
    # --------------------- #
    # freq
    # --------------------- #
    "log_freq_epoch": 1000,
    "plot_freq_epoch": 1000,
    "save_freq_epoch": 1000,
    "eval_freq_epoch": 1000,
    # --------------------- #
}
# =============================== #

parallel_processes = []
#=============================== #

# num_cuda_avail = torch.cuda.device_count()
gpu = 0
run_config = {
        "model": "SameClassDelta",
 }
    # =============================== #

config = DEFAULT_CONFIG
config.update(run_config)
arg_str = get_arg_string(config)
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

print("Running arguments: [{}]".format(arg_str))


run_command = "{} ./test_corr_x_xprime_cosine.py {}".format(PYTHON_EXE, arg_str).strip()

subprocess.call(run_command, shell=True)