import torch
import torch.nn as nn
import numpy as np
import random

from utils.models import LiteBaseModel
from utils.shaping import flatten_right_from

from torchvision.utils import save_image

import torchvision.transforms.functional as F

import torchvision.transforms as T


class SameClassDelta(LiteBaseModel):
    def __init__(self, classifier, data_augmenter, target_class, prob,
                 input_shape, num_trainable_samples=128, norm_type="l1",
                 input_postprocessing_fn=lambda x: x,
                 device='cuda'):

        LiteBaseModel.__init__(self, device)

        self.add_module('classifier', classifier)
        #self.classifier = classifier
        if data_augmenter is None:
            data_augmenter = nn.Identity()
        self.add_module('data_augmenter', data_augmenter)

        assert isinstance(target_class, int), \
            f"target_class must be an integer. " \
            f"Found type(target_class)={type(target_class)}!"
        self.target_class = target_class
        self.prob = prob

        assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 3, \
            f"'input_shape' must be a list/tuple length 3. " \
            f"Found input_shape={input_shape}!"
        self.input_shape = input_shape
        c, w, h = tuple(input_shape)

        assert isinstance(num_trainable_samples, int) and num_trainable_samples > 0, \
            f"num_trainable_samples={num_trainable_samples}!"
        self.num_trainable_samples = num_trainable_samples

        # Reversed samples
        # --------------------------- #
        self.original_images = None
        self.original_labels = None
        self.pred_labels = None
        self._trainable_param = torch.randn(
            (num_trainable_samples, c, w, h), requires_grad=True,
            device=self.device, dtype=torch.float32)
        self.trainable_param_inited = False
        # --------------------------- #

        assert norm_type in ('l1', 'l2'), \
            f"norm_type must be in ('l1', 'l2'). Found {norm_type}!"
        self.norm_type = norm_type

        assert callable(input_postprocessing_fn), \
            "'input_processing_fn' must be callable!"
        self.input_postprocessing_fn = input_postprocessing_fn

        self._xent_criterion = nn.CrossEntropyLoss(reduction='none')

        print(f"In class [{self.__class__}]")
        print(f"target_class: {self.target_class}")
        print(f"input_shape: {self.input_shape}")
        print(f"num_trainable_samples: {self.num_trainable_samples}")
        print(f"trainable_param_inited: {self.trainable_param_inited}")
        print(f"norm_type: {self.norm_type}")
        print(f"device: {self.device}")

    # Eval and train has no effect to the classifier
    def eval(self):
        old_training = self.training
        self.training = False

        return old_training

    def train(self, mode=True):
        old_training = self.training
        self.training = mode

        return old_training

    def save(self, file_path, initpath, include_teacher=False, **kwargs):
        # print ("OH SAMECLASSDELTA SAVE FUNCTION!!!!")
        save_obj = dict()

        C = self.classifier
        if hasattr(C, 'module') and C.module is not None:
            C = C.module
        save_obj['classifier_state_dict'] = C.cpu().state_dict()
        C.to(self.device)

        save_obj['_trainable_param'] = self._trainable_param.cpu()
        ## save the final x+\del_x as numpy array here.
        np.save(initpath+"/initial_images_plusdelx",self.trainable_images.detach().cpu())
        torch.save(save_obj, file_path, **kwargs)

    def load(self, file_path, include_teacher=False, **kwargs):
        save_obj = torch.load(file_path, **kwargs)

        C = self.classifier

        if hasattr(C, 'module') and C.module is not None:
            C = C.module

        C.load_state_dict(save_obj[f'classifier_state_dict'])

        self._trainable_param = save_obj[f'_trainable_param'].to(self.device)

    ## No transformation is done on the trainable images
    ## Everytime it is as it is.
    @property
    def trainable_images(self):
        return self._trainable_param
 
    def init_trainable_samples_pure(self, images, labels, initdir=None):
        assert labels.dtype == torch.long and labels.shape == (images.shape[0],), \
            f"labels.dtype={labels.dtype}, labels.shape={labels.shape}, images.shape={images.shape}!"
        assert images.shape == self._trainable_param.shape, \
            f"images.shape={images} while " \
            f"self._trainable_param.shape={self._trainable_param.shape}!"
        self.original_labels = labels.clone().detach().to(self.device)
        self.original_images = images.clone().detach().to(self.device) 
        self._trainable_param.data.copy_(self.original_images)
        self.trainable_param_inited = True
        logit_org = self.postproc_and_classify(self.original_images)
        probs_org = torch.softmax(logit_org, dim=-1)
        pred_prob_org, pred_labels = torch.max(probs_org, dim=-1)
        if initdir!=None:
            # save the initial image and the labels 
            np.save(initdir+'/initial_images_org',self.original_images.detach().cpu())
            np.save(initdir+'/pred_labels',pred_labels.detach().cpu())

    # based on the predicted class
    def init_trainable_samples_pure_basedonpred(self, images, labels, initdir=None):
        # print (image.shape)
        assert labels.dtype == torch.long and labels.shape == (images.shape[0],), \
            f"labels.dtype={labels.dtype}, labels.shape={labels.shape}, images.shape={images.shape}!"
        assert images.shape == self._trainable_param.shape, \
            f"images.shape={images} while " \
            f"self._trainable_param.shape={self._trainable_param.shape}!"
        self.original_labels = labels.clone().detach().to(self.device)
        self.original_images = images.clone().detach().to(self.device) ##changes reflect in both?
        logit_org = self.postproc_and_classify(self.original_images)
        probs_org = torch.softmax(logit_org, dim=-1)
        pred_prob_org, pred_labels = torch.max(probs_org, dim=-1)
        self.pred_labels = pred_labels.clone().detach().to(self.device)
        self._trainable_param.data.copy_(self.original_images)
        self.trainable_param_inited = True
        logit_org = self.postproc_and_classify(self.original_images)
        probs_org = torch.softmax(logit_org, dim=-1)
        pred_prob_org, pred_labels = torch.max(probs_org, dim=-1)
        if initdir!=None:
            # save the initial image and the labels 
            np.save(initdir+'/initial_images_org',self.original_images.detach().cpu())
            np.save(initdir+'/pred_labels',pred_labels.detach().cpu())


    def classify(self, x, y=None):
        y_logit = self.postproc_and_classify(x)
        y_prob = torch.softmax(y_logit, dim=-1)
        y_pred = torch.argmax(y_logit, dim=-1)

        if y is not None:
            xent = self._xent_criterion(y_logit, y).mean(0)
            xent = xent.data.cpu().item()

            acc = y_pred.eq(y).to(torch.float32).mean(0)
            acc = acc.data.cpu().item()
        else:
            xent = None
            acc = None

        return {
            'y_prob': y_prob.data,
            'y_pred': y_pred.data,
            'acc': acc,
            'xent': xent,
        }

    def get_all_train_params(self, *args, **kwargs):
        return [self._trainable_param]

    def postproc_and_classify(self, x):
        return self.classifier(self.input_postprocessing_fn(x))
    

    def get_loss(self, ids, loss_coeffs):
        assert self.trainable_param_inited, \
            f"'trainable_param' must be initialized in advance!"
        lc = loss_coeffs
        x_org = self.original_images[ids]
        ## self.trainable_images two cases one is to set within 0 and 1, other let it go anywhere.
        x = self.trainable_images[ids]
        t = self.target_class * torch.ones_like(ids)
        y_logit = self.postproc_and_classify(self.data_augmenter(x))
        ## divided softmax by Temperature
        y_logit =  y_logit/lc['T']
        xent = self._xent_criterion(y_logit, t).mean(0)
        ###prob loss
        norm_reg = torch.norm(flatten_right_from(x - x_org, dim=1),
                              p=(1 if self.norm_type == 'l1' else 2),
                              dim=1).mean(0)
        
        loss = lc['xent'] * xent + lc['norm_reg'] * norm_reg

        return {
            "loss": loss,
            "xent": xent.data.cpu().item(),
            "norm_reg": norm_reg.data.cpu().item(),
        }
    def get_set_loss(self, ids, loss_coeffs):
        assert self.trainable_param_inited, \
            f"'trainable_param' must be initialized in advance!"
        lc = loss_coeffs
        x_org = self.original_images[ids]
        ## self.trainable_images two cases one is to set within 0 and 1, other let it go anywhere.
        x = self.trainable_images[ids]
        ## SET wrong CLASS
        t = 8 * torch.ones_like(ids)
        y_logit = self.postproc_and_classify(self.data_augmenter(x))
        ## divided softmax by Temperature
        y_logit =  y_logit/lc['T']
        xent = self._xent_criterion(y_logit, t).mean(0)
        ###prob loss
        norm_reg = torch.norm(flatten_right_from(x - x_org, dim=1),
                              p=(1 if self.norm_type == 'l1' else 2),
                              dim=1).mean(0)
        
        loss = lc['xent'] * xent + lc['norm_reg'] * norm_reg

        return {
            "loss": loss,
            "xent": xent.data.cpu().item(),
            "norm_reg": norm_reg.data.cpu().item(),
        }
    ## x' changes with respect to the predicted ie class (by the classifier)
    # useful in unknown label setting
    def get_loss_basedon_prediction(self, ids, loss_coeffs):
        assert self.trainable_param_inited, \
            f"'trainable_param' must be initialized in advance!"
        lc = loss_coeffs
        x_org = self.original_images[ids]
        x = self.trainable_images[ids] 
        t = self.pred_labels[ids] 
        y_logit = self.postproc_and_classify(self.data_augmenter(x))
        y_logit =  y_logit/lc['T']
        xent = self._xent_criterion(y_logit, t).mean(0)
        norm_reg = torch.norm(flatten_right_from(x - x_org, dim=1),
                              p=(1 if self.norm_type == 'l1' else 2),
                              dim=1).mean(0)
        
        loss = lc['xent'] * xent + lc['norm_reg'] * norm_reg 

        return {
            "loss": loss,
            "xent": xent.data.cpu().item(),
            "norm_reg": norm_reg.data.cpu().item(),
        }
 
def save_noisy_image(img, name):
    if img.size(1) == 3:
        img = img.view(img.size(0), 3, 32, 32)
        save_image(img, name)
    else:
        img = img.view(img.size(0), 1, 28, 28)
        save_image(img, name)

def inv_sigmoid(x, eps=1e-6):
    # x = 1/(1 + e^-y)
    # y = log(x) - log(1-x)

    x = torch.clamp(x, eps, 1 - eps)
    y = x.log() - (1 - x).log()
    return y

def clip_image(x):
    x = torch.clamp(x, min=0, max=1)
    return x

## Before performing TanH on the trainable inputs
def inv_sigmoid_via_tanh(x, eps=1e-6):
    x = torch.clamp(2 * x - 1, -1 + eps, 1 - eps)
    y = torch.arctanh(x)
    return y
