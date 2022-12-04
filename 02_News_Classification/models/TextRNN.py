import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config:
    "配置参数"
    def __init__(self, dataset, embedding):
        self.model_name = "TextRNN"
        self.train_path = dataset + "/data/train.txt"
        self.dev_path = dataset + "data/dev.txt"
        self.test_path = dataset + "data/test.txt"
        self.class_list = [x.strip() for x in open(dataset + "/data/class.txt").readlines()]



class Model(nn.Module):
    def __init__(self, config):
        ...
