# coding: utf-8
import sys
sys.path.append('..')
import os
import pickle
import numpy
import pprint
from ch08.attention_seq2seq import AttentionSeq2seq
from common import config, trainer, optimizer
from common.util import eval_seq2seq, to_gpu
from dataset import sequence
from hyperparam import get_neurons_size, get_max_grad

class Char2IdSeq2Seq:
    def __init__(self, trainFile, char2idFile=None, modelFile=None, 
                 sprit_ratio=10, sep='_'):
        self.sep = sep
        if trainFile is not None:
            file_path = os.path.dirname(os.path.abspath(__file__))
            print('trainFile file_path: ', file_path)
            train, test, c = Char2IdSeq2Seq.load_data(trainFile, file_path, 
                                                      sprit_ratio, sep)
            self.x_train, self.t_train = train[0], train[1]
            self.x_test, self.t_test = test[0], test[1]
            self.char2id, self.id2char = c[0], c[1]
            self.quest_size = len(self.x_train[0])
            self.cor_size = len(self.t_train[0]) - 1
            print('self.quest_size:', self.quest_size,
                  ' self.cor_size:', self.cor_size)

        if char2idFile is not None:
            self.char2id, self.id2char, self.sep = Char2IdSeq2Seq.load_char2id(char2idFile)

        vocab_size = len(self.char2id)
        wordvec_size ,hidden_size = get_neurons_size()
        self.seq2seq = AttentionSeq2seq(vocab_size, wordvec_size ,hidden_size)
        self.trainer = trainer.Trainer(self.seq2seq, optimizer=optimizer.Adam())
        if modelFile is not None:
            self.load_params(modelFile)

    def fit(self, max_epoch=10, batch_size=32, eval_interval=20):
        self.trainer.fit(self.x_train, self.t_train, max_epoch, 
                         batch_size, get_max_grad(), eval_interval)

    def generate(self, xs, start_id, sample_size):
        return self.seq2seq.generate(xs, start_id, sample_size)

    def predict(self, st, sample_size):
            st = self.c2i(st)
            st = numpy.array([st]) # numpyかつ、(1,x)に変換
            st = st[:, ::-1]
            guess = self.generate(st, self.char2id[self.sep], sample_size)
            return self.i2c(guess)

    def i2c(self, ids):
        return ''.join([self.id2char[int(ch)] for ch in ids])

    def c2i(self, sts):
        return [self.char2id[ch] if ch in self.char2id else self.char2id[' '] for ch in sts]

    def display_accuracy_rate(self):
        correct_num = 0
        for i in range(len(self.x_test)):
            q, c = self.x_test[[i]], self.t_test[[i]]
            correct_num += eval_seq2seq(self.seq2seq, q, c, self.id2char, is_reverse=True)
        acc = float(correct_num) / len(self.x_test)
        print('val acc %.3f%%' % (acc * 100))

    def save_params(self, modelFile=None, char2idFile=None):
        if char2idFile is None:
            char2idFile = 'char2id.pkl'
        with open(char2idFile, 'wb') as f:
            pickle.dump({'char2id': self.char2id, 
                         'id2char': self.id2char,
                         'sep': self.sep
                         }, f)
        self.seq2seq.save_params(modelFile)

    def load_params(self, modelFile=None):
        self.seq2seq.load_params(modelFile)

    @staticmethod
    def load_data(trainFile, file_path, sprit_ratio, sep):
        (x_train, t_train), (x_test, t_test) = sequence.load_data(
            trainFile, 
            sep=sep, 
            file_path=file_path, 
            sprit_ratio=sprit_ratio)
        char2id, id2char = sequence.get_vocab()
        
        x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
        
        if config.GPU:
            print("Training Mode GPU(config.GPU): ", config.GPU)
            x_train, t_train = to_gpu(x_train), to_gpu(t_train)
            x_test, t_test = to_gpu(x_test), to_gpu(t_test)
        
        return (x_train, t_train), (x_test, t_test), (char2id, id2char)

    @staticmethod
    def load_char2id(char2idFile=None):
        if char2idFile is None:
            char2idFile = 'char2id.pkl'
        if '/' in char2idFile:
            char2idFile = char2idFile.replace('/', os.sep)
        if not os.path.exists(char2idFile):
            raise IOError('No file: ' + char2idFile)
        with open(char2idFile, 'rb') as f:
            data = pickle.load(f)
        return data['char2id'], data['id2char'], data['sep']

    @staticmethod
    def display_char2id(char2idFile=None):
        c2i, i2c = Char2IdSeq2Seq.load_char2id(char2idFile)
        pprint(c2i, i2c)
