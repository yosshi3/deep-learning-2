# coding: utf-8
"""
'text_domain.txt'でトレーニングする
"""
import char2id_trainer

char2id_trainer.train('text_domain.txt', 
                      sprit_ratio=50,
                      max_epoch=10,
                      modelFile='domain_model.pkl', 
                      char2idFile='domain_char2id.pkl')
