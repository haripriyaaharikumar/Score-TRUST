# from six import iteritems
import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetContainer(Dataset):
    def __init__(self, base_dataset):
        assert isinstance(base_dataset, Dataset), "'base_dataset' " \
            "must be an instance of Dataset. Found {}!".format(type(Dataset))
        self.base_dataset = base_dataset
        # self.__dict__.update(base_dataset.__dict__)

    def __getattr__(self, attr):
        if hasattr(self.base_dataset, attr):
            return getattr(self.base_dataset, attr)
        else:
            raise AttributeError(f"Cannot find attribute '{attr}' for class '{self.__class__}'!")



class DataSubset(DatasetContainer):
    def __init__(self, base_dataset, size_or_ids, seed=None):
        super(DataSubset, self).__init__(base_dataset)

        if isinstance(size_or_ids, int):
            size = size_or_ids
            assert size < len(base_dataset), "'base_dataset' only have {} samples " \
                "while 'size_or_ids'={}!".format(len(base_dataset), size)
            self.ids = np.random.RandomState(seed).choice(
                list(range(len(base_dataset))), size, replace=False)
        else:
            assert hasattr(size_or_ids, '__len__')
            self.ids = size_or_ids

    def __getitem__(self, idx):
        base_idx = self.ids[idx]
        return self.base_dataset[base_idx]

    def __len__(self):
        return len(self.ids)


class DatasetWithTransform(DatasetContainer):
    def __init__(self, base_dataset, transform):
        super(DatasetWithTransform, self).__init__(base_dataset)

        assert hasattr(base_dataset, 'transform'), "base_dataset must have the attribute 'transform'!"
        assert base_dataset.transform is None, f"base_dataset.transform must be None!"

        assert callable(transform), f"'transform' is not callable!"
        self.transform = transform

    def __getitem__(self, idx):
        output = self.base_dataset[idx]
        x = self.transform(output[0])
        return (x, ) + output[1:]

    def __len__(self):
        return len(self.base_dataset)



class DatasetContainer_custom(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getattr__(self, attr):
        if hasattr(self.base_dataset, attr):
            return getattr(self.base_dataset, attr)
        else:
            raise AttributeError(f"Cannot find attribute '{attr}' for class '{self.__class__}'!")


class DataSubset_woDataset_typecheck(DatasetContainer_custom):
    def __init__(self, base_dataset, size_or_ids, seed=None):
       self.ids = size_or_ids
       print (f"THE LENGTH-----{len(size_or_ids)}")
       self.base_dataset = base_dataset

    def __getitem__(self, idx):
        base_idx = self.ids[idx]
        return self.base_dataset[base_idx]

    def __len__(self):
        return len(self.ids)
  
class MyDataset(Dataset):
    def __init__(self, data, target, transform=None, target_transform=None):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        
        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)
            
        return x, y
    
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
    
class DataSubset_woDataset_typecheck(DatasetContainer_custom):
    def __init__(self, base_dataset, size_or_ids, seed=None):
       self.ids = size_or_ids
       print (f"THE LENGTH-----{len(size_or_ids)}")
       self.base_dataset = base_dataset

    def __getitem__(self, idx):
        base_idx = self.ids[idx]
        return self.base_dataset[base_idx]

    def __len__(self):
        return len(self.ids)
 