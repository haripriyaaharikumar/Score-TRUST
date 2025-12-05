import os
import torch
import torch.nn as nn


# Very useful for rapid deployment
class LiteBaseModel:
    def __init__(self, device='cuda'):
        self.device = device
        self.training = True
        self.modules = []
        self.module_names = []

    def stop_grad(self):
        # Stop gradients of all modules of this model
        for m in self.modules:
            m.requires_grad_(False)

    def add_module(self, module_name, module):
        assert isinstance(module, torch.nn.Module), \
            f"'{module_name}' must be a " \
            f"torch.nn.Module. Found {type(module)}!"

        self.__setattr__(module_name, module)
        self.modules.append(module)
        self.module_names.append(module_name)

    def print_modules(self):
        print("Modules of {}:".format(self.__class__))
        for i in range(len(self.module_names)):
            print("{}: {}".format(self.module_names[i], self.modules[i].__class__))

    def get_module_params(self, module_name):
        return self.__getattribute__(module_name).parameters()

    def train(self, mode=True):
        for module in self.modules:
            module.train(mode)

        old_training = self.training
        self.training = mode

        return old_training

    def eval(self):
        for module in self.modules:
            module.eval()

        old_training = self.training
        self.training = False
        return old_training
    
    def to(self, device):
        for m in self.modules:
            m.to(device)

    def save(self, file_path, *args, **kwargs):
        # print ("BASE SAVE FUNCTION!!!")
        save_obj = dict()

        for i in range(len(self.modules)):
            mn = self.module_names[i]
            mp = self.modules[i]

            if hasattr(mp, 'module') and mp.module is not None:
                mp = mp.module

            mp.to('cpu')
            save_obj[f'{mn}_state_dict'] = mp.state_dict()
            mp.to(self.device)

        torch.save(save_obj, file_path, *args, **kwargs)

    def save_dir(self, folder_path, *args, **kwargs):
        self.save(os.path.join(folder_path, "model.pt"), *args, **kwargs)

    def save_dir_wdinit(self, folder_path, initpath, *args, **kwargs):
        self.save(os.path.join(folder_path, "model.pt"), initpath, *args, **kwargs)

    def load(self, file_path, *args, **kwargs):
        save_obj = torch.load(file_path, *args, **kwargs)

        for i in range(len(self.modules)):
            mn = self.module_names[i]
            mp = self.modules[i]

            if hasattr(mp, 'module') and mp.module is not None:
                mp = mp.module

            mp.load_state_dict(save_obj[f'{mn}_state_dict'])

    def load_dir(self, folder_path, *args, **kwargs):
        self.load(os.path.join(folder_path, "model.pt"), *args, **kwargs)