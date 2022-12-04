import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description="Chinese News Classification")
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == "__main__":
    dataset = "THUCNews"
    # 搜狗新闻: embedding_SougouNews.npz 腾讯新闻: embedding_Tencent.npz 随机: random
    embedding = "embedding_SougouNews.npz"
    if args.embedding == "random":
        embedding = "random"
    model_name = args.model
    # print(model_name)
    if model_name == "FastText":
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = "random"
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module("models." + model_name)
    config = x.Config(dataset, embedding)

