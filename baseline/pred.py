import logging
import pickle
import copy

from get_data import prepare_data
from config import batching_list_instances
from train_model import load_model


def pred(model, dev_insts, config):
    model.eval()
    batch_size = 32
    dev_batches = batching_list_instances(config, dev_insts)
    batch_id = 0
    f_w = open("pred_result.txt", "w", encoding="utf8")
    for batch in dev_batches:
        one_batch_insts = dev_insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        input_ids, input_seq_lens, annotation_mask, labels = batch
        input_masks = input_ids.gt(0)
        batch_max_scores, batch_max_ids = model(input_ids, input_seq_lens=input_seq_lens, annotation_mask=annotation_mask,
                         labels=None, attention_mask=input_masks)
        batch_id += 1
        for i in range(len(input_ids)):
            query_list = one_batch_insts[i].input.words
            label_tensor = batch_max_ids[i].cpu().numpy()[:len(query_list)][::-1]
            label_list = [config.idx2labels[k] for k in label_tensor]
            for q,l in zip(query_list, label_list):
                f_w.write(q + " " + l +"\n")
            f_w.write("\n")


def main():

    config_name = "/data/xlxia/code/AutoIE/baseline/saved_model/config.conf"
    with open(config_name, 'rb') as f:
        config = pickle.load(f)
    label2idx = copy.deepcopy(config.label2idx)

    _, devs = prepare_data(logging, config)
    config.label2idx = label2idx
    model = load_model(config, 'final_bert_crf/best.pth.tar')
    pred(model, devs, config)


if __name__ == "__main__":
    main()
