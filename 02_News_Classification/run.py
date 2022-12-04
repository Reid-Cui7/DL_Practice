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

    # 设定随机种子
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)




