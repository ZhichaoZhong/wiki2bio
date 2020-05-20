
import sys
import os
import tensorflow as tf
import time
from wiki2bio.SeqUnit import *

import tensorflow as tf
import numpy as np
from wiki2bio.DataLoader import DataLoader
from wiki2bio.PythonROUGE import PythonROUGE
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from wiki2bio.preprocess import *
from wiki2bio.util import *
WORK_DIR = "/Users/zzhong/PycharmProjects/table2text/data/if_data_0518"
hidden_size = 500
emb_size = 400
field_size = 50
pos_size = 5
batch_size = 32
epoch = 2
source_vocab = 20003
field_vocab = 1480
position_vocab = 31
target_vocab = 20000
field_vocab = 177
position_vocab = 31
target_vocab = 20000
report = 5000
learning_rate = 0.0003
mode = 'train'
load_ = "0"
dir_ = f"{WORK_DIR}/processed_data/"
limits = 0

dual_attention = True
fgate_encoder = True
field = False
position = False
encoder_pos = True
decoder_pos = True
#
# tf.app.flags.DEFINE_integer("hidden_size", 500, "Size of each layer.")
# tf.app.flags.DEFINE_integer("emb_size", 400, "Size of embedding.")
# tf.app.flags.DEFINE_integer("field_size", 50, "Size of embedding.")
# tf.app.flags.DEFINE_integer("pos_size", 5, "Size of embedding.")
# tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size of train set.")
# tf.app.flags.DEFINE_integer("epoch", 50, "Number of training epoch.")
# tf.app.flags.DEFINE_integer("source_vocab", 20003,'vocabulary size')
# tf.app.flags.DEFINE_integer("field_vocab", 1480,'vocabulary size')
# tf.app.flags.DEFINE_integer("position_vocab", 31,'vocabulary size')
# tf.app.flags.DEFINE_integer("target_vocab", 20003,'vocabulary size')
# tf.app.flags.DEFINE_integer("report", 5000,'report valid results after some steps')
# tf.app.flags.DEFINE_float("learning_rate", 0.0003,'learning rate')
#
# tf.app.flags.DEFINE_string("mode",'train','train or test')
# tf.app.flags.DEFINE_string("load",'0','load directory') # BBBBBESTOFAll
# tf.app.flags.DEFINE_string("dir", f'{WORK_DIR}/processed_data/', 'data set directory')
# tf.app.flags.DEFINE_integer("limits", 0,'max data set size')
#
#
# tf.app.flags.DEFINE_boolean("dual_attention", True,'dual attention layer or normal attention')
# tf.app.flags.DEFINE_boolean("fgate_encoder", True,'add field gate in encoder lstm')
#
# tf.app.flags.DEFINE_boolean("field", False,'concat field information to word embedding')
# tf.app.flags.DEFINE_boolean("position", False,'concat position information to word embedding')
# tf.app.flags.DEFINE_boolean("encoder_pos", True,'position information in field-gated encoder')
# tf.app.flags.DEFINE_boolean("decoder_pos", True,'position information in dual attention decoder')
#

last_best = 0.0
gold_path_test = f'{WORK_DIR}/processed_data/test/test_split_for_rouge/gold_summary_'
gold_path_valid = f'{WORK_DIR}/processed_data/valid/valid_split_for_rouge/gold_summary_'

# test phase
if load_ != "0":
    save_dir = f'{WORK_DIR}/results/res/' + load_ + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = f'{WORK_DIR}/results/evaluation/' + load_ + '/'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'
# train phase
else:
    prefix = str(int(time.time() * 1000))
    save_dir = f'{WORK_DIR}/results/res/' + prefix + '/'
    save_file_dir = save_dir + 'files/'
    pred_dir = f'{WORK_DIR}/results/evaluation/' + prefix + '/'
    os.mkdir(save_dir)
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(save_file_dir):
        os.mkdir(save_file_dir)
    pred_path = pred_dir + 'pred_summary_'
    pred_beam_path = pred_dir + 'beam_summary_'

log_file = save_dir + 'log.txt'



def train(sess, dataloader, model):
    write_log("#######################################################")
    # for flag in __flags:
    #     write_log(flag + " = " + str(__flags[flag]))
    write_log("#######################################################")
    trainset = dataloader.train_set
    k = 0
    loss, start_time = 0.0, time.time()
    for _ in range(epoch):
        for x in dataloader.batch_iter(trainset, batch_size, True):
            loss += model(x, sess)
            k += 1
            progress_bar(k % report, report)
            if (k % report == 0):
                cost_time = time.time() - start_time
                write_log("%d : loss = %.3f, time = %.3f " % (k // report, loss, cost_time))
                loss, start_time = 0.0, time.time()
                if k // report >= 1:
                    ksave_dir = save_model(model, save_dir, k // report)
                    write_log(evaluate(sess, dataloader, model, ksave_dir, 'valid'))


def test(sess, dataloader, model):
    evaluate(sess, dataloader, model, save_dir, 'test')


def save_model(model, save_dir, cnt):
    new_dir = save_dir + 'loads' + '/'
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    nnew_dir = new_dir + str(cnt) + '/'
    if not os.path.exists(nnew_dir):
        os.mkdir(nnew_dir)
    model.save(nnew_dir)
    return nnew_dir


def evaluate(sess, dataloader, model, ksave_dir, mode='valid'):
    if mode == 'valid':
        texts_path = f"{WORK_DIR}/processed_data/valid/valid.box.val"
        gold_path = gold_path_valid
        evalset = dataloader.dev_set
    else:
        texts_path = f"{WORK_DIR}/processed_data/test/test.box.val"
        gold_path = gold_path_test
        evalset = dataloader.test_set

    # for copy words from the infoboxes
    texts = open(texts_path, 'r').read().strip().split('\n')
    texts = [list(t.strip().split()) for t in texts]
    v = Vocab()

    # with copy
    pred_list, pred_list_copy, gold_list = [], [], []
    pred_unk, pred_mask = [], []

    k = 0
    for x in dataloader.batch_iter(evalset, batch_size, False):
        predictions, atts = model.generate(x, sess)
        atts = np.squeeze(atts)
        idx = 0
        for summary in np.array(predictions):
            with open(pred_path + str(k), 'w') as sw:
                summary = list(summary)
                if 2 in summary:
                    summary = summary[:summary.index(2)] if summary[0] != 2 else [2]
                real_sum, unk_sum, mask_sum = [], [], []
                for tk, tid in enumerate(summary):
                    if tid == 3:
                        sub = texts[k][np.argmax(atts[tk, : len(texts[k]), idx])]
                        real_sum.append(sub)
                        mask_sum.append("**" + str(sub) + "**")
                    else:
                        real_sum.append(v.id2word(tid))
                        mask_sum.append(v.id2word(tid))
                    unk_sum.append(v.id2word(tid))
                sw.write(" ".join([str(x) for x in real_sum]) + '\n')
                pred_list.append([str(x) for x in real_sum])
                pred_unk.append([str(x) for x in unk_sum])
                pred_mask.append([str(x) for x in mask_sum])
                k += 1
                idx += 1
    write_word(pred_mask, ksave_dir, mode + "_summary_copy.txt")
    write_word(pred_unk, ksave_dir, mode + "_summary_unk.txt")

    for tk in range(k):
        with open(gold_path + str(tk), 'r') as g:
            gold_list.append([g.read().strip().split()])

    gold_set = [[gold_path + str(i)] for i in range(k)]
    pred_set = [pred_path + str(i) for i in range(k)]

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_list)
    copy_result = "with copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
                  (str(F_measure), str(recall), str(precision), str(bleu))
    # print copy_result

    for tk in range(k):
        with open(pred_path + str(tk), 'w') as sw:
            sw.write(" ".join(pred_unk[tk]) + '\n')

    recall, precision, F_measure = PythonROUGE(pred_set, gold_set, ngram_order=4)
    bleu = corpus_bleu(gold_list, pred_unk)
    nocopy_result = "without copy F_measure: %s Recall: %s Precision: %s BLEU: %s\n" % \
                    (str(F_measure), str(recall), str(precision), str(bleu))
    # print nocopy_result
    result = copy_result + nocopy_result
    # print result
    if mode == 'valid':
        print(result)

    return result


def write_log(s):
    print(s)
    with open(log_file, 'a') as f:
        f.write(s + '\n')


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

copy_file(save_file_dir)
dataloader = DataLoader(dir_, limits)
# copy_file(save_file_dir)
with tf.Session(config=config) as sess:

    model = SeqUnit(batch_size=batch_size, hidden_size=hidden_size, emb_size=emb_size,
                    field_size=field_size, pos_size=pos_size, field_vocab=field_vocab,
                    source_vocab=source_vocab, position_vocab=position_vocab,
                    target_vocab=target_vocab, scope_name="seq2seq", name="seq2seq",
                    field_concat=field, position_concat=position,
                    fgate_enc=fgate_encoder, dual_att=dual_attention, decoder_add_pos=decoder_pos,
                    encoder_add_pos=encoder_pos, learning_rate=learning_rate)

    sess.run(tf.global_variables_initializer())
    if load_ != '0':
        model.load(save_dir)
    if mode == 'train':
        train(sess, dataloader, model)
    else:
        test(sess, dataloader, model)
