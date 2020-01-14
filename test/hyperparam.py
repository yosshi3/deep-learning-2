# coding: utf-8
"""
ハイパーパラメータ
"""
def get_neurons_size():
    # wordvec_size = 100
    wordvec_size = 16
    hidden_size = 128
    return wordvec_size, hidden_size

def get_max_grad():
    max_grad = 5.0
    return max_grad

# def get_sep():
#     sep = '\t'
#     return sep

def get_qsize():
    qsize=50
    return qsize

def get_csize():
    csize=1
    return csize

