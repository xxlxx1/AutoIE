import argparse
import numpy as np
from typing import Tuple, List
import time
import torch
from common import Instance
import os
import logging
import pickle

import itertools
from torch.optim import Adam
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from transformers import BertConfig
from bert_model import BertCRF
import utils
from argument import parse_arguments_t
from get_data import prepare_data, hard_constraint_predict, set_seed
from config import Config, evaluate_batch_insts, batching_list_instances


def train_model(config: Config, train_insts: List[List[Instance]], dev_insts: List[Instance]):
    train_num = sum([len(insts) for insts in train_insts])
    logging.info(("[Training Info] number of instances: %d" % (train_num)))
    # get the batched data
    dev_batches = batching_list_instances(config, dev_insts)

    model_folder = config.model_folder
    logging.info("[Training Info] The model will be saved to: %s" % (model_folder))
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    num_outer_iterations = config.num_outer_iterations

    for iter in range(num_outer_iterations):

        logging.info("-" * 20 + f" [Training Info] Running for {iter}th large iterations. " + "-" * 20)

        model_names = []  # model names for each fold
        train_batches = [batching_list_instances(config, insts) for insts in train_insts]

        # train 2 models in 2 folds
        for fold_id, folded_train_insts in enumerate(train_insts):
            logging.info("\n" + f"-------- [Training Info] Training fold {fold_id}. Initialized from pre-trained Model -------")
            model_name = model_folder + f"/bert_crf_{fold_id}"
            model_names.append(model_name)
            train_one(config=config, train_batches=train_batches[fold_id],  # Initialize bert model
                      dev_insts=dev_insts, dev_batches=dev_batches, model_name=model_name)

        # assign prediction to other folds
        logging.info("\n\n[Data Info] Assigning labels")

        # using the model trained in one fold to predict the result of another fold's data
        # and update the label of another fold with the predict result
        for fold_id, folded_train_insts in enumerate(train_insts):

            cfig_path = os.path.join(config.bert_model_dir, 'bert_config.json')
            cfig = BertConfig.from_json_file(cfig_path)
            cfig.device = config.device
            cfig.label2idx = config.label2idx
            cfig.label_size = config.label_size
            cfig.idx2labels = config.idx2labels

            model_name = model_folder + f"/bert_crf_{fold_id}"
            model = BertCRF(cfig=cfig)
            model.to(cfig.device)
            utils.load_checkpoint(os.path.join(model_name, 'best.pth.tar'), model)

            hard_constraint_predict(config=config, model=model,
                                    fold_batches=train_batches[1 - fold_id],
                                    folded_insts=train_insts[1 - fold_id])  # set a new label id, k is set to 2, so 1 - fold_id can be used

        # train the final model
        logging.info("\n\n")
        logging.info("-------- [Training Info] Training the final model-------- ")

        # merge the result data to training the final model
        all_train_insts = list(itertools.chain.from_iterable(train_insts))

        logging.info("Initialized from pre-trained Model")
        model_name = model_folder + "/final_bert_crf"
        config_name = model_folder + "/config.conf"
        all_train_batches = batching_list_instances(config=config, insts=all_train_insts)
        train_one(config=config, train_batches=all_train_batches, dev_insts=dev_insts, dev_batches=dev_batches,
                          model_name=model_name, config_name=config_name)
        # load the best final model
        # utils.load_checkpoint(os.path.join(model_name, 'best.pth.tar'), model)
        # model.eval()
        # logging.info("\n")
        # result = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        logging.info("\n\n")


def train_one(config: Config, train_batches: List[Tuple], dev_insts: List[Instance], dev_batches: List[Tuple],
              model_name: str, config_name: str = None) -> BertCRF:

    # load config for bertCRF
    cfig_path = os.path.join(config.bert_model_dir,
                             'bert_config.json')
    cfig = BertConfig.from_json_file(cfig_path)
    cfig.device = config.device
    cfig.label2idx = config.label2idx
    cfig.label_size = config.label_size
    cfig.idx2labels = config.idx2labels
    # load pretrained bert model
    model = BertCRF.from_pretrained(config.bert_model_dir, config=cfig)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(config.device)

    if config.full_finetuning:
        logging.info('full finetuning')
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

    else:
        logging.info('tuning downstream layer')
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

    optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.05 * epoch))

    model.train()

    epoch = config.num_epochs
    best_dev_f1 = -1
    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()

        for index in np.random.permutation(len(train_batches)):  # disorder the train batches
            model.train()
            scheduler.step()
            input_ids, input_seq_lens, annotation_mask, labels = train_batches[index]
            input_masks = input_ids.gt(0)
            # update loss
            loss = model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                         labels=labels, attention_mask=input_masks)
            epoch_loss += loss.mean().item()
            model.zero_grad()
            loss.mean().backward()
            # gradient clipping
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            optimizer.step()
        end_time = time.time()
        logging.info("Epoch %d: loss:%.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time))

        model.eval()
        with torch.no_grad():
            # metric is [precision, recall, f_score]
            dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
            if dev_metrics[2] > best_dev_f1:  # save the best model
                # logging.info(" " * 90 + "saving the best model...")
                best_dev_f1 = dev_metrics[2]

                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                optimizer_to_save = optimizer
                utils.save_checkpoint({'epoch': epoch + 1,
                                       'state_dict': model_to_save.state_dict(),
                                       'optim_dict': optimizer_to_save.state_dict()},
                                      is_best=dev_metrics[2] > 0,
                                      checkpoint=model_name)

                # Save the corresponding config as well.
                if config_name:
                    f = open(config_name, 'wb')
                    pickle.dump(config, f)
                    f.close()
        model.zero_grad()
    # return model


def evaluate_model(config: Config, model: BertCRF, batch_insts_ids, name: str, insts: List[Instance]):
    # evaluation
    metrics = np.asarray([0, 0, 0], dtype=int)
    batch_id = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]

        input_ids, input_seq_lens, annotation_mask, labels = batch
        input_masks = input_ids.gt(0)
        # get the predict result
        batch_max_scores, batch_max_ids = model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                         labels=None, attention_mask=input_masks)

        metrics += evaluate_batch_insts(batch_insts=one_batch_insts,
                                        batch_pred_ids=batch_max_ids,
                                        batch_gold_ids=batch[-1],
                                        word_seq_lens=batch[1], idx2label=config.idx2labels)
        batch_id += 1
    # calculate the precision, recall and f1 score
    p, total_predict, total_entity = metrics[0], metrics[1], metrics[2]
    precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    logging.info(" " * 42 + "[%s set] Precision: %.2f, Recall: %.2f, F1: %.2f" % (name, precision, recall, fscore))
    return [precision, recall, fscore]


def main():
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
    logging.info("batch size:" + str(conf.batch_size))

    trains, devs = prepare_data(logging, conf, opt)
    train_model(config=conf, train_insts=trains, dev_insts=devs)


if __name__ == "__main__":
    main()
