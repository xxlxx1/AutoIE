import torch
import random
import math
import numpy as np

from typing import List, Tuple

from config import Reader, Config
from bert_model import BertCRF
from common import Instance


def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def prepare_data(logging, conf):
    conf.train_file = conf.dataset + "/train.txt"
    conf.dev_file = conf.dataset + "/valid.txt"
    # data reader
    reader = Reader(conf.digit2zero)

    # read trains/devs
    logging.info("\n")
    logging.info("Loading the datasets...")
    trains = reader.read_txt(conf.train_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)

    logging.info("Building label idx ...")
    conf.build_label_idx(trains + devs)

    random.shuffle(trains)
    # set the prediction flag, if is_prediction is False, we will not update this label.
    for inst in trains:
        inst.is_prediction = [False] * len(inst.input)
        for pos, label in enumerate(inst.output):
            if label == conf.O:
                inst.is_prediction[pos] = True
    # dividing the data into 2 parts(num_folds default to 2)
    num_insts_in_fold = math.ceil(len(trains) / conf.num_folds)
    trains = [trains[i * num_insts_in_fold: (i + 1) * num_insts_in_fold] for i in range(conf.num_folds)]
    return trains, devs


def hard_constraint_predict(config: Config, model: BertCRF, fold_batches: List[Tuple], folded_insts:List[Instance], model_type:str = "hard"):
    """using the model trained in one fold to predict the result of another fold"""
    batch_id = 0
    batch_size = config.batch_size
    model.eval()
    for batch in fold_batches:
        one_batch_insts = folded_insts[batch_id * batch_size:(batch_id + 1) * batch_size]

        input_ids, input_seq_lens, annotation_mask, labels = batch
        input_masks = input_ids.gt(0)
        # get the predict result
        batch_max_scores, batch_max_ids = model(input_ids, input_seq_lens=input_seq_lens,
                                                annotation_mask=annotation_mask, labels=None, attention_mask=input_masks)

        batch_max_ids = batch_max_ids.cpu().numpy()
        word_seq_lens = batch[1].cpu().numpy()
        for idx in range(len(batch_max_ids)):
            length = word_seq_lens[idx]
            prediction = batch_max_ids[idx][:length].tolist()
            prediction = prediction[::-1]
            # update the labels of another fold
            one_batch_insts[idx].output_ids = prediction
        batch_id += 1