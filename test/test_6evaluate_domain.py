# coding: utf-8
"""
'text_train.txt'の全データを評価する。間違いがあればprintする。
"""
import char2id_trainer

char2id_trainer.evaluate('text_domain.txt',
                         modelFile='domain_model.pkl',
                         char2idFile='domain_char2id.pkl')
