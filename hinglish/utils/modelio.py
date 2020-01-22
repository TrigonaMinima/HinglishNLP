import dill
import torch


def load_torch_pickle(path):
    return torch.load(path, pickle_module=dill)
