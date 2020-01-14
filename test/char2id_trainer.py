# coding: utf-8
"""
char2id_seq2seq専用のtrainer
"""
import sys
sys.path.append('..')
from char2id_seq2seq import Char2IdSeq2Seq

def train(trainFile,
          sep='_',
          sprit_ratio=10, 
          batch_size = 256,
          max_epoch = 30,
          eval_test_interval = 5, # 複数echoch間のテストデータ評価間隔
          eval_loss_interval = 25, # 1epoch内のバッチ回数を対象とした損失評価間隔
          char2idFile='char2id.pkl',
          modelFile='char2idModel.pkl'):
    """
    seq2seqの訓練（サンプル評価有り）
    """
    model = Char2IdSeq2Seq(trainFile, sprit_ratio=sprit_ratio, sep=sep)
    
    for epoch in range(max_epoch // eval_test_interval):
        model.fit(eval_test_interval, batch_size, eval_loss_interval)
        model.display_accuracy_rate()

    model.save_params(modelFile, char2idFile)


def evaluate(trainFile, char2idFile='char2id.pkl', modelFile='char2idModel.pkl'):
    """
    seq2seqの評価
    トレーニングデータの全てを評価する。間違いがあればprintする。
    """
    model = Char2IdSeq2Seq(None, char2idFile, modelFile)

    with open(trainFile, 'r', encoding="utf-8") as f:
        quests = [line.split(model.sep) for line in f]

    c_num = 0
    for i, q in enumerate(quests):
        print("i=", i) if i % 500 == 0 else None
        guess = model.predict(q[0], 1) # 予測(サイズ1)
        c = q[1].strip() #空白と改行を除去
        if(c != guess):
            print("cor:", c, " guess:", guess, " quest:", q[0])
        c_num += 1 if guess == c else 0
    acc = float(c_num) / len(quests)
    print('val acc %.3f%%' % (acc * 100))


def predict(predictFile, resultFile, 
            char2idFile='char2id.pkl', modelFile='char2idModel.pkl'):
    """
    seq2seqの予測
    """
    model = Char2IdSeq2Seq(None, char2idFile, modelFile)

    with open(predictFile, 'r', encoding="utf-8") as f:
        quests = [line.split(model.sep) for line in f]
    
    with open(resultFile, 'w', encoding='utf-8', newline='\n') as f:
        for i, q in enumerate(quests):
            print("i=", i) if i % 500 == 0 else None
            guess = model.predict(q[0], 1) # 予測(サイズ1)
            line = "{0:50}{1}{2}".format(q[0], model.sep, guess)
            f.writelines(line + '\n')
