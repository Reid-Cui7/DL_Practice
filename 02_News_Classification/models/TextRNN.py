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
        self.vocab_path = dataset + "/data/vocab.pkl" # 词表
        self.save_path = dataset + "/saved_dict/" + self.model_name + ".ckpt"
        self.log_path = dataset + "/log/" + self.model_name
        self.embedding_pretrained = torch.tensor(np.load(dataset + "/data/" + embedding)["embeddings"].astype("float32"))\
                if embedding != "random" else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dropout = 0.5
        self.require_improvement - 1000
        self.num_classes = len(self.class_list)
        self.n_vocab = 0 # 词表大小 运行时赋值
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size - 32 # 每句话处理成的长度 短填长切
        self.learning_rate = 1e-3
        self.embed = self.embedding_pretrained.size(1) if self.embedding_pretrained else 300 # 字向量维度
        self.hidden_size = 128 # LSTM隐藏层
        self.num_layers = 2

class Model(nn.Module):
    def __init__(self, config):
        ...
