# coding: utf-8

from hyperparam import get_sep
import char2id_trainer

char2id_trainer.predict('text_test.txt', 'text_predict.txt')

# 'text_predict.txt'からdomain分だけを抽出
with open('text_predict_true_only.txt', 'w', encoding='utf-8', newline='\n') as f2:
    for line in open('text_predict.txt', 'r', encoding="utf-8"):
        strs = line.split(get_sep())
        if(strs[1][0] == '1'):
                f2.writelines(str(strs[0]) + '\n')
