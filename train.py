from __future__ import print_function

import os
from os.path import join
import json
import argparse
from time import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from utils.general_str import make_dir_if_not_exist
from utils.arg_parsing import str2bool
from utils.training import BestResultsTracker, SaveDirTracker
from utils.training import ScalarSummarizer
from utils.utils_manager import *

from models import SameClassDelta
from models import pureclsfr


parser = argparse.ArgumentParser(allow_abbrev=False)


# Dataset
# ---------------------------------- #
parser.add_argument('--dataset', type=str, required=True,
                    choices=['cifar10','camelyon17', 'tinyimagenet', 'imagenet'])
parser.add_argument('--data_root', required=True, type=str)
parser.add_argument('--workers', default=2, type=int)

parser.add_argument('--train_seed', default=1234, type=int)

# Augmentations
# ---------------------------------- #
parser.add_argument('--posttrigger_augment', default='False', type=str2bool)
parser.add_argument('--random_crop', default=5, type=int)
parser.add_argument('--random_rotation', default=10, type=int)

# ---------------------------------- #
# Model
# ---------------------------------- #
parser.add_argument('--model', default='SameClassDelta')
parser.add_argument('--net', default='pre_resnet18', type=str,
                    choices=['pre_resnet18', 'netC_MNIST', 'resnet50', 'vgg11', 'simplenet', 'ViT'])
parser.add_argument('--arch', default='vinai', type=str, choices=['vinai'])
# ---------------------------------- #

# Training
# ---------------------------------- #
parser.add_argument('--epochs', default=600, type=int)
parser.add_argument('--batch_size', default=128, type=int)
# ---------------------------------- #

# Optimizers
# ---------------------------------- #
parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam', 'lbfgs'])
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--nesterov', type=str2bool, default='False')
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--schedule_lr', type=str2bool, default='False')
parser.add_argument('--lr_milestones', type=int, nargs="+", default=[])
parser.add_argument('--lr_decay', type=float, default=1.0)
# ---------------------------------- #

# Settings
# ---------------------------------- #
parser.add_argument('--target_class', type=int, required=True)
parser.add_argument('--probability', type=float, required=True)
parser.add_argument('--norm_type', type=str, default='l1', choices=['l1', 'l2'])
parser.add_argument('--use_tanh', type=str2bool, default='True')

# ---------------------------------- #
# Coefficients
# ---------------------------------- #
parser.add_argument('--xent', type=float, default=1.0)
parser.add_argument('--norm_reg', type=float, default=0.0)
parser.add_argument('--prob_reg', type=float, default=0.0)
parser.add_argument('--tv_reg', type=float, default=0.0)
parser.add_argument('--T', type=float, default=5.0)
# ---------------------------------- #

# Saved model
# ---------------------------------- #
parser.add_argument('--model_output_dir', type=str, required=True)
parser.add_argument('--model_load_step', type=int, required=True)
parser.add_argument('--model_load_best', type=str2bool, required=True)
# ---------------------------------- #

# Log
# ---------------------------------- #
parser.add_argument('--log_freq', type=int, default=1)
parser.add_argument('--plot_freq', type=int, default=1)
parser.add_argument('--save_freq_epoch', type=int, default=1)
parser.add_argument('--eval_freq_epoch', type=int, default=1)
# ---------------------------------- #

# Save/Resume
# ---------------------------------- #
parser.add_argument('--max_save', type=int, default=2)
parser.add_argument('--max_save_best', type=int, default=2)
# New config that will overwrite old config when resuming
# ---------------------------------- #

# Run
# ---------------------------------- #
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
# ---------------------------------- #


class ModelConfig(object):
    pass


class TrainEnvironment(object):
    pass


def train(args, env):
    model = env.model
    optim = env.optim
    schl = env.schl

    train_ids_sampler = env.train_ids_sampler
    train_summarizer = env.train_summarizer

    save_tracker = env.save_tracker
    save_best_tracker = env.save_best_tracker
    best_results_tracker = env.best_results_tracker

    model.train()
    for b in range(env.steps_per_epoch):
        env.global_step += 1

        # Get data based  on the ids
        # --------------------------------------- #
        ids = train_ids_sampler.sample_ids()
        ids = torch.tensor(np.asarray(ids, np.int32),
                           dtype=torch.long, device=env.device)
        # --------------------------------------- #
        # Optimization based on the Classifier Prediction classes
        # --------------------------------------- #
        batch_results = model.get_loss_basedon_prediction(ids=ids, loss_coeffs=env.loss_coeffs)

        loss = batch_results['loss']

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Accumulate results
        # --------------------------------------- #
        train_summarizer.accumulate(batch_results, ids.shape[0])
        # --------------------------------------- #

    # Update LR
    # --------------------- #
    if schl is not None:
        schl.step()
 
    if env.epoch % args.log_freq == 0:
        log_time_end = time()
        log_time_gap = (log_time_end - env.log_time_start)
        env.log_time_start = log_time_end

        _, train_results = train_summarizer.get_summaries_and_reset(summary_prefix='train')

        log_str = "\n[Train][{}][{}][{}], Epoch {}/{}, Step {} ({:.2f}s)".format(
            args.dataset, args.model, args.run, env.epoch, args.epochs, env.global_step, log_time_gap) + \
            "\n" + ", ".join(["{}: {:.4f}".format(key, train_results[key]) for key in env.train_keys])

        with open(env.train_log_file, "a") as f:
            f.write(log_str)
            f.write("\n")
        f.close()
    # ---------------------------------- #
    # Save model and train_state
    # ---------------------------------- #
    if env.global_step % args.log_freq == 0:
        print("Save at global_step={}!".format(env.global_step))
        save_dir = make_dir_if_not_exist(save_tracker.get_save_dir(env.global_step))
        model.save_dir_wdinit(save_dir,env.initdata)

        save_tracker.update_and_delete_old_save(env.global_step)

        train_state = {
            'global_step': env.global_step,
            'epoch': env.epoch + 1,

            'optimizer': optim.state_dict(),
            'scheduler': None if schl is None else schl.state_dict(),

            'steps': save_tracker.get_saved_steps(),
            'steps_best': save_best_tracker.get_saved_steps(),
            'best_results': best_results_tracker.get_best_results(),
        }
        torch.save(train_state, join(save_dir, "train_state.pt"))


def main(args):
    # Create environment variable
    # ===================================== #
    env = TrainEnvironment()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env.device = device

    np.set_printoptions(suppress=True, precision=3, threshold=np.inf, linewidth=1000)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=1000)

    ## OUTPUT DIR for results
    args.output_dir = join(args.output_dir, args.net + "_" + args.arch,
                           args.model,args.optim,str(args.epochs) +
                           "_lr" +str(args.lr)+ "_norm"+args.norm_type+"reg"
                           +str(args.norm_reg)+"_Temp"
                           +str(args.T),args.run+str(args.target_class))


    args_model = ModelConfig()
    config_file = join(args.model_output_dir, 'config.json')
    with open(config_file, 'r') as f:
        config_model = json.load(f)
    args_model.__dict__.update(config_model)
    # ===================================== #
    # Load data
    # ===================================== #
    dataset_manager = get_dataset_manager_test(args)
    env.train_dataset = dataset_manager['test_dataset']
    env.train_ids_sampler = dataset_manager['test_dataset_id_sampler']
    train_images = dataset_manager['test_images']
    train_labels = dataset_manager['test_labels']
    num_train_samples = len(train_labels)
    print ("TRAIN SAMPLES...",num_train_samples)
    assert 0 <= args.batch_size <= num_train_samples, \
        f"batch_size={args.batch_size} while " \
        f"num_train_samples={num_train_samples}!"

    env.steps_per_epoch = num_train_samples // args.batch_size
    if args.log_freq < 0:
        args.log_freq = env.steps_per_epoch

    if args.plot_freq < 0:
        args.plot_freq = env.steps_per_epoch
    # ===================================== #

    # Load model
    # ===================================== #

    baseclassifier = get_component(args)
    data_augmenter = get_data_augmenter(args)
    if data_augmenter is not None:
        data_augmenter = data_augmenter.to(device)

    baseclassifier = pureclsfr(args=args, classifier=baseclassifier,
     return_features=False, num_classes=args.num_classes, device=device)

    num_gpus = torch.cuda.device_count()
    print(f"num_gpus: {num_gpus}")
    if num_gpus > 1:
        baseclassifier.classifier = torch.nn.DataParallel(baseclassifier.classifier)
        cudnn.benchmark = True
    baseclassifier.classifier = baseclassifier.classifier.to(device)
    if args.model_load_best:
        model_save_dir = join(args_model.output_dir, "model",
                                  f"best_step_{args.model_load_step}")
    else:
        model_save_dir = join(args_model.output_dir, "model",
                                  f"step_{args.model_load_step}")

    baseclassifier.load_dir(model_save_dir)
    baseclassifier.stop_grad()
    baseclassifier.eval() ###

    if args.model == "SameClassDelta":
        model = SameClassDelta(baseclassifier.classifier,data_augmenter=data_augmenter,
            target_class=args.target_class, prob=args.probability,
            input_shape=args.input_shape,
            num_trainable_samples=train_labels.shape[0],
            norm_type=args.norm_type,
            device=device)

    else:
        raise ValueError(f"Do not support args.model={args.model}!")
    env.initdata = make_dir_if_not_exist(join(args.output_dir,"initdata"))  
   
    # # ## LABELS WILL BE INITIATED AFTER ADDING NOISE
    model.init_trainable_samples_pure_basedonpred(images=train_images, labels=train_labels,initdir=env.initdata)

    env.model = model
    
    loss_coeffs = {
        'xent': args.xent,
        'norm_reg': args.norm_reg,
        'T':args.T,
    }
    env.loss_coeffs = loss_coeffs
    # ===================================== #

    # To device
    # ===================================== #
    num_gpus = torch.cuda.device_count()
    print(f"num_gpus: {num_gpus}")
    if num_gpus > 1:
        cudnn.benchmark = True

        # print("Use data parallel!")
        model.classifier = torch.nn.DataParallel(model.classifier)
    print ("Model loaded successfully")
    model.classifier = model.classifier.to(device)
    baseclassifier = baseclassifier.to(device)


    # ===================================== #

    # Build optimizer
    # ===================================== #
    print("\nGet optimizers!")
    optim, schl = get_optimizer(args, model)
    env.optim = optim
    env.schl = schl
    # ===================================== #

    # Create directories
    # ===================================== #
    asset_dir = make_dir_if_not_exist(join(args.output_dir, "asset"))
    env.train_img_dir = make_dir_if_not_exist(join(asset_dir, "train_img"))
    env.test_img_dir = make_dir_if_not_exist(join(asset_dir, "test_img"))

    log_dir = make_dir_if_not_exist(join(args.output_dir, "log"))
    env.train_log_file = join(log_dir, "train.log")
    env.test_log_file = join(log_dir, "test.log")

    model_dir = make_dir_if_not_exist(join(args.output_dir, "model"))

    env.model_dir = model_dir
    # ===================================== #

    # Summarizers and Trackers
    # ===================================== #
    # Summarizer

    env.train_keys = ['loss', 'xent', 'norm_reg']

    env.train_summarizer = ScalarSummarizer([(key, 'mean') for key in env.train_keys])

    # Tracker
    save_tracker = SaveDirTracker(args.max_save, dir_path_prefix=join(model_dir, "step_"))
    save_best_tracker = SaveDirTracker(args.max_save_best, dir_path_prefix=join(model_dir, "best_step_"))
    best_results_tracker = BestResultsTracker([('acc', 'greater')], num_best=args.max_save_best)

    env.save_tracker = save_tracker
    env.save_best_tracker = save_best_tracker
    env.best_results_tracker = best_results_tracker

    start_epoch = 0
    # ===================================== #
    # Training
    # ===================================== #

    env.log_time_start = time()
    print ("start the training process")
    for epoch in range(start_epoch, args.epochs):
        ### env.epoch =  each epoch number
        env.epoch = epoch
        # --------------------- #
        # Train
        # --------------------- #
        train(args, env)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
