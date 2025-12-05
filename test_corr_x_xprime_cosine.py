
from __future__ import print_function

import os
from os.path import join
import json
import argparse
from time import time

import torch.backends.cudnn as cudnn
from utils.training import get_oldest_save_dir
from utils.general_str import make_dir_if_not_exist
from utils.arg_parsing import  str2bool
from models import SameClassDelta
from models import pureclsfr
from scipy import spatial

import utils.utils_manager as utils_manager
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


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
                    choices=['pre_resnet18', 'netC_MNIST', 'vgg11', 'simplenet', 'ViT'])
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
parser.add_argument('--norm_type', type=str, default='l1', choices=['l1', 'l2'])
parser.add_argument('--use_tanh', type=str2bool, default='True')


# ---------------------------------- #
# Coefficients
# ---------------------------------- #
parser.add_argument('--xent', type=float, default=1.0)
parser.add_argument('--norm_reg', type=float, default=0.0)
# ---------------------------------- #

# Saved TRUST model
# ---------------------------------- #
parser.add_argument('--trust_model_output_dir', type=str, required=True)
parser.add_argument('--trust_model_load_step', type=int, required=True)
parser.add_argument('--trust_model_load_best', type=str2bool, required=True)
# ---------------------------------- #


# Saved classifier model
# ---------------------------------- #
parser.add_argument('--model_output_dir', type=str, required=True)
parser.add_argument('--model_load_step', type=int, required=True)
parser.add_argument('--model_load_best', type=str2bool, required=True)
# ---------------------------------- #
# Save
# --------------------------------- #
parser.add_argument('--max_save', type=int, default=2)
parser.add_argument('--max_save_best', type=int, default=2)

# ---------------------------------- #

# Run
# ---------------------------------- #
parser.add_argument('--output_dir', type=str, required=True)



class TrainEnvironment(object):
    pass

def detn_dataset_feature_basedon_pred(args, env):
    model = env.model
    basemodel = env.baseclassifier ##pure classifier 
    model.eval()
    with torch.no_grad():
        transformed_img = model.trainable_images.data
        org_img = model.original_images
        logit_org, intmd_ft1 = basemodel.postproc_and_classify(org_img)
        probs_orgs = torch.softmax(logit_org, dim=-1)
        _, pred_class = torch.max(probs_orgs, dim=-1)
        transformed_img = model.trainable_images.data
        _, intmd_ft2 = basemodel.postproc_and_classify(transformed_img)
    return intmd_ft1, intmd_ft2, pred_class


def trust_score(args):
    # Create environment variable
    # ===================================== #
    env = TrainEnvironment()
    # ===================================== #

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') ## is this assigning GPU 0 only???
    env.device = device

    np.set_printoptions(suppress=True, precision=3, threshold=np.inf, linewidth=1000)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=1000)
 
    if args.dataset == "cifar10":
        # clss =list(range(0,10)) ## add target class at the end
        clss = [0]

    elif args.dataset =="gtsrb":
        clss = list(range(0,43))

   
    #----------------------------------------------------#
    ##   Per class, intermediate TRUST score save
    #----------------------------------------------------#

    for i in range(len(clss)): 
            args.target_class = clss[i]  
            l=f"0_class{clss[i]}"  
            dir_l = join("_trainall","test_dataset")
            dataset_manager = utils_manager.get_dataset_manager_test(args)
            env.train_dataset = dataset_manager['test_dataset']
            env.train_ids_sampler = dataset_manager['test_dataset_id_sampler']
            train_images = dataset_manager['test_images']
            train_labels = dataset_manager['test_labels']
            baseclassifier = utils_manager.get_component(args)
            data_augmenter = utils_manager.get_data_augmenter(args)
            ## Load the saved x+delx model
            trust_model_output_dir1 = join(args.trust_model_output_dir+dir_l,l)
            baseclassifier = pureclsfr(args=args, classifier=baseclassifier,
                return_features=True, num_classes=args.num_classes, device=device)

            baseclassifier.classifier = baseclassifier.classifier.to(device)
            # Load saved Trust model either based on the best model or the given step
            # ----------------------------- #
            if args.model_load_best:
                model_save_dir = join(args.model_output_dir, "model",
                                        f"best_step_{args.model_load_step}")
            else:
                model_save_dir = join(args.model_output_dir, "model",
                                        f"step_{args.model_load_step}")


            baseclassifier.load_dir(model_save_dir) ## Base model
            baseclassifier.stop_grad()
            baseclassifier.eval()
            env.baseclassifier = baseclassifier
            if args.model == "SameClassDelta":
                    model = SameClassDelta(baseclassifier.classifier,data_augmenter=data_augmenter,
                        target_class=args.target_class, prob=1.0,
                        input_shape=args.input_shape,
                        num_trainable_samples=train_labels.shape[0],
                        norm_type=args.norm_type,
                        device=device)
            ## -- fine to assign pred_labels as train_labels since its not using for loss computation
            model.init_trainable_samples_pure(images=train_images, labels=train_labels)

            # Loading saved model
            if args.trust_model_load_best:
                if args.trust_model_load_step >= 0:
                    save_dir = join(trust_model_output_dir1, "model",
                    f"best_step_{args.trust_model_load_step}")
                else:
                    save_dir = get_oldest_save_dir(join(trust_model_output_dir1, "model"),
                        "best_step_")
            else:
                save_dir = get_oldest_save_dir(join(trust_model_output_dir1,"model"), 
                        "step_")

            model.load_dir(save_dir)
            env.model = model
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                cudnn.benchmark = True
                model.classifier = torch.nn.DataParallel(model.classifier)

            model.classifier = model.classifier.to(device)

            env.curr_cls = clss[i] ## assign initial class w.r.t train datasets to pick same class data
            intmdt_ftr1, intmdt_ftr2, all_labels = detn_dataset_feature_basedon_pred(args, env)
            
            all_labels_np = all_labels.cpu().numpy()
            intmdt_ftr1_np = intmdt_ftr1.cpu().numpy()

            if len(intmdt_ftr1_np.shape)>2: 
                tot_f1, ch_f1, r_f1, c_f1 = intmdt_ftr1_np.shape ## for x
                intmdt_ftr2_np = intmdt_ftr2.cpu().numpy()
                tot_f2, ch_f2, r_f2, c_f2 = intmdt_ftr2_np.shape ## for x'
                intmdt_ftr1_train_flat = intmdt_ftr1_np.reshape(-1,ch_f1*r_f1*c_f1)
                intmdt_ftr2_train_flat = intmdt_ftr2_np.reshape(-1,ch_f2*r_f2*c_f2)
            else:
                tot_f1, ch_f1 = intmdt_ftr1_np.shape ## for x
                intmdt_ftr2_np = intmdt_ftr2.cpu().numpy()
                tot_f2, ch_f2 = intmdt_ftr2_np.shape ## for x'
                intmdt_ftr1_train_flat = intmdt_ftr1_np.reshape(-1,ch_f1)
                intmdt_ftr2_train_flat = intmdt_ftr2_np.reshape(-1,ch_f2)               
            corrs = [] 
            for ii in range(tot_f1):
                corr_each = 1 - spatial.distance.cosine(intmdt_ftr1_train_flat[ii], intmdt_ftr2_train_flat[ii])
                corrs += [corr_each] ## for mmd to make it each element
            corrs_allclass_same_pred_np = np.array(corrs) 

            env.test_img_dir = make_dir_if_not_exist(join("/..", "TRUST","test")) ## Layer-wise analysis on CIFAR-10
            ## class-wise TRUST score between x and x+del_x
            np.save(env.test_img_dir+f"/trust_test_class_{clss[i]}.npy",corrs_allclass_same_pred_np)
            ### Predicted class
            np.save(env.test_img_dir+f"/test_class_{clss[i]}_pred_labels.npy", all_labels_np)

    ## Merge all into a csv file
    perdata = 1000 # for CIFAR-10, vary depend on dataset
    all_cos_test = []
    all_pred_test = []
    all_true_test = []
    for i in range(len(clss)): 
        cos_test = np.load(env.test_img_dir+f"cosine_sim_test_class_{i}.npy")
        pred_test = np.load(env.test_img_dir+f"test_class_{i}_pred_labels.npy")
        all_cos_test+=cos_test.tolist()
        all_pred_test+=pred_test.tolist()
        all_true_test+=[i]*perdata

    df_test = pd.DataFrame({"Trust":all_cos_test,
                                "Org class": all_true_test,
                                "Pred class": all_pred_test})

    df_test.to_csv(f'Test_cosine_cifar10.csv')


if __name__ == '__main__':
    args = parser.parse_args()
    trust_score(args)
    print ("done")