
import sys
import os
import tensorflow as tf
import time
from wiki2bio.SeqUnit import *
from wiki2bio.DataLoader import DataLoader
import numpy as np
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

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

copy_file(save_file_dir)
dataloader = DataLoader(dir_, limits)
model = SeqUnit(batch_size=batch_size, hidden_size=hidden_size, emb_size=emb_size,
                field_size=field_size, pos_size=pos_size, field_vocab=field_vocab,
                source_vocab=source_vocab, position_vocab=position_vocab,
                target_vocab=target_vocab, scope_name="seq2seq", name="seq2seq",
                field_concat=field, position_concat=position,
                fgate_enc=fgate_encoder, dual_att=dual_attention, decoder_add_pos=decoder_pos,
                encoder_add_pos=encoder_pos, learning_rate=learning_rate)

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
# copy_file(save_file_dir)
if load_ != '0':
    model.load(save_dir)
if mode == 'train':
    train(sess, dataloader, model)
else:
    test(sess, dataloader, model)

sess.close()