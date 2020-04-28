import os
import logging
import argparse
import pickle

from transformers import BertConfig
from bert_model import BertCRF
import utils
from argument import parse_arguments_t
from get_data import prepare_data
from config import Config, batching_list_instances


def load_model(config):
    cfig_path = os.path.join(config.bert_model_dir, 'bert_config.json')
    cfig = BertConfig.from_json_file(cfig_path)
    cfig.device = config.device
    cfig.label2idx = config.label2idx
    cfig.label_size = config.label_size
    cfig.idx2labels = config.idx2labels

    model = BertCRF(cfig=cfig)
    model.to(cfig.device)
    utils.load_checkpoint(os.path.join(config.model_folder, 'final_bert_crf/best.pth.tar'), model)
    model.eval()
    return model


def pred(model, dev_insts, config):
    batch_size = 32
    dev_batches = batching_list_instances(config, dev_insts)
    batch_id = 0
    for batch in dev_batches:
        one_batch_insts = dev_insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        input_ids, input_seq_lens, annotation_mask, labels = batch
        input_masks = input_ids.gt(0)
        batch_max_scores, batch_max_ids = model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                         labels=None, attention_mask=input_masks)
        print()


def main():
    logging.info("Transformer implementation")
    parser = argparse.ArgumentParser(description="Transformer CRF implementation")
    # opt = parse_arguments_t(parser)
    # conf = Config(opt)

    # set logger
    # utils.set_logger(os.path.join("log", opt.log_name))

    # params
    # for k in opt.__dict__:
    #     logging.info(k + ": " + str(opt.__dict__[k]))
    # logging.info("batch size:" + str(conf.batch_size))
    config_name = "/data/xlxia/code/AutoIE/baseline/saved_model/config.conf"
    with open(config_name, 'rb') as f:
        config = pickle.load(f)
    label2idx = config.label2idx

    _, devs = prepare_data(logging, config)
    config.label2idx = label2idx
    model = load_model(config)
    pred(model, devs, config)


if __name__ == "__main__":
    main()
