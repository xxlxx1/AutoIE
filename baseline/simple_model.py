import argparse
from typing import List
from common import Instance
import os
import logging

import utils
from argument import parse_arguments_t
from get_data import prepare_data, set_seed
from config import Config, batching_list_instances
from train_model import train_one


def train_model(config: Config, train_insts: List[List[Instance]], dev_insts: List[Instance]):
    train_num = sum([len(insts) for insts in train_insts])
    logging.info(("[Training Info] number of instances: %d" % (train_num)))
    dev_batches = batching_list_instances(config, dev_insts)   # 验证集一直不会改变

    model_folder = config.model_folder
    logging.info("[Training Info] The model will be saved to: %s" % (model_folder))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    logging.info("-" * 20 + f" [Training Info] Running for {iter}th large iterations. " + "-" * 20)
    train_batches = [batching_list_instances(config, insts) for insts in train_insts]

    logging.info("\n" + f"-------- [Training Info] Training fold {0}. Initialized from pre-trained Model -------")
    model_name = model_folder + f"/bert_crf_simple"
    train_one(config=config, train_batches=train_batches[0],  # Initialize bert model
                  dev_insts=dev_insts, dev_batches=dev_batches, model_name=model_name)


if __name__ == "__main__":
    logging.info("Transformer implementation")
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    opt = parse_arguments_t(parser)
    conf = Config(opt)
    set_seed(opt, conf.seed)
    # set logger
    utils.set_logger(os.path.join("log", opt.log_name))

    # params
    for k in opt.__dict__:
        logging.info(k + ": " + str(opt.__dict__[k]))

    trains, devs = prepare_data(logging, conf)
    train_model(config=conf, train_insts=trains, dev_insts=devs)

