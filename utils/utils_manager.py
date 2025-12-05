import random
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as tr, datasets as ds
import kornia.augmentation as aug
import numpy as np

from utils import extract_data_by_total_counts
from utils.training import ContinuousIndexSampler
from utils.sampler import ContinuousBatchSampler
from utils import DataSubset

from models import PreActResNet18
import torchvision

class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTriggerAugmenter(torch.nn.Module):
    def __init__(self, args):
        super(PostTriggerAugmenter, self).__init__()

        h = args.input_shape[1]
        w = args.input_shape[2]

        self.random_crop = ProbTransform(aug.RandomCrop((h, w), padding=args.random_crop), p=0.8)
        self.random_rotation = ProbTransform(aug.RandomRotation(args.random_rotation), p=0.5)
        if args.dataset == "cifar10":
            self.random_horizontal_flip = aug.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

def get_transform(args, train=True):
    if args.dataset in ("cifar10", "gtsrb"):
        h = w = 32
    else:
        raise ValueError("Do not support dataset={args.dataset}!")

    transforms = list()
    transforms.append(tr.Resize((h, w)))

    if train:
        transforms.append(tr.RandomCrop((h, w), padding=args.random_crop))
        if args.dataset == "cifar10": 
            transforms.append(tr.RandomHorizontalFlip(p=0.5))
    transforms.append(tr.ToTensor())

    return tr.Compose(transforms)


def get_dataset_manager_test(args):
    train_transform = get_transform(args, train=False)
    test_transform = get_transform(args, train=False)
    if args.dataset == "cifar10":
        args.num_classes = 10
        args.input_shape = (3, 32, 32)
        train_dataset = ds.CIFAR10(args.data_root, train=True,
                                   transform=train_transform, download=True)
        test_dataset = ds.CIFAR10(args.data_root, train=False,
                                  transform=test_transform, download=True)
        test_labels = test_dataset.targets

    
    else:
        raise ValueError(f"Do not support dataset={args.dataset}!")
    test_dataset_tarids = []
    test_images = []
    test_labelss = []
    for i in range(len(test_labels)):
        if test_dataset.targets[i] == args.target_class:
            test_dataset_tarids.append(i)
            x, y = test_dataset[i]
            test_images.append(x)
            test_labelss.append(y)

    test_dataset_tarids = np.array(test_dataset_tarids)
    test_dataset_sub = DataSubset(test_dataset, test_dataset_tarids)
    test_images = torch.stack(test_images, dim=0)
    test_labelss = torch.from_numpy(np.asarray(test_labelss, dtype=np.int32)).to(torch.long)


    batch_sampler_d = ContinuousBatchSampler(
        len(test_dataset), num_repeats=5, 
        batch_size=args.batch_size, shuffle=True)

    test_dataset_loader = DataLoader(
        test_dataset, batch_sampler=batch_sampler_d,
        num_workers=args.workers, pin_memory=True)

    test_dataset_loader = iter(test_dataset_loader)

    test_dataset_id_sampler = ContinuousIndexSampler(
        len(test_dataset_tarids), sample_size=args.batch_size, shuffle=True)

    test_dataset_loader = DataLoader(test_dataset_sub, batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True,
                             shuffle=False, drop_last=False)
    return {
        'test_dataset': test_dataset_sub,
        'test_images': test_images,
        'test_labels': test_labelss,
        'test_dataset_id_sampler': test_dataset_id_sampler,
        'test_dataset_loader': test_dataset_loader,
    }

def get_component(args):
    # classifier
    # ------------------------ #
    if args.dataset == "cifar10":
        if args.net == "pre_resnet18":
            classifier = PreActResNet18(num_classes=10)
    else:
        raise Exception(f"Invalid dataset '{args.dataset}'!")
    return classifier


def get_data_augmenter(args):
    return None


def get_optimizer(args, model):
    params = model.get_all_train_params()
    print(f"len(params): {len(params)}")
    print(f"params[0].shape: {params[0].shape}")

    if args.optim == "sgd":
        print("Use SGD optimizer!")
        optim = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                nesterov=args.nesterov,
                                weight_decay=args.weight_decay)
    elif args.optim == "adam":
        print("Use Adam optimizer!")
        optim = torch.optim.Adam(params, lr=args.lr,
                                 betas=(args.beta1, args.beta2),
                                 weight_decay=args.weight_decay)

    elif args.optim == "lbfgs":
        print("Use LBFGS optimizer!")
        optim = torch.optim.LBFGS(params, lr=args.lr)

    else:
        raise ValueError(f"Do not support optim={args.optim}!")

    if args.schedule_lr:
        schl = torch.optim.lr_scheduler.MultiStepLR(
            optim, args.lr_milestones, args.lr_decay)
    else:
        schl = None

    return optim, schl


