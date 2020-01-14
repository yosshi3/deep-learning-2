# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 02:20:36 2020

@author: aaa
"""

import re
import pprint

f1 = open('text_predict_true_only.txt', 'r', encoding='utf-8', newline='\n')
f2 = open('text_domain.txt', 'w', encoding='utf-8', newline='\n')

domain_vaules=[]
for line in f1:
    domain_vaules.append(line)

# 「△: 」の場合、「:」の前後の空白は除去を打ち消す。例）「△: 1:売」を「△:」「1:売」に分解するため
re1 = re.compile('△: ')
# 「:」の前後の空白は除去
re2 = re.compile(' : |: | :')
# 「'(」「' (」の間に「:」を追加
re3 = re.compile("'\(|' \(")
# 「…」を「:」に変換
re4 = re.compile("…")
# 「'|‘|’|”|“|″」を「"」に統一
re5 = re.compile("'|‘|’|”|“|″")
# 「"1":休日」、「「1」:営業日」 のカッコを除去
re6 = re.compile(r'"(.+?)":|「(.+?)」:')
# 「休日:"1"」の後ろカッコを除去
re7 = re.compile(r':"(.+?)"')
# 空白、「、」「,」を区切り文字として分割
re8 = re.compile('、|,')
re9 = re.compile('\s+')

re10 = re.compile('^"|"$')

re11 = re.compile('[0-9a-zA-Z△]')

def cleansing1(x):
    x = re1.sub('△:  ', x)
    x = re2.sub(':', x)
    x = re3.sub("':(", x)
    x = re4.sub(':', x)
    x = re5.sub('"', x)
    x = re6.sub(r'\1:', x)
    x = re7.sub(r':\1', x)
    x = re8.sub(" ", x)
    x = re9.split(x)
    return x

def cleansing2(x):
    x = re10.sub('', x)
    return x


domain_vaules = list(map(lambda x: cleansing1(x), domain_vaules))
domain_vaules = sum(domain_vaules, []) # 2次元配列をflattenする
domain_vaules = list(map(lambda x: cleansing2(x), domain_vaules))
domain_vaules = list(set(domain_vaules)) #重複排除

domain_vaules.sort()
pprint.pprint(domain_vaules[:10])

for i in range(len(domain_vaules)):
    sentence = domain_vaules[i]
    pattern = 0
    if(sentence.count(':') == 1):
        if(re11.match(sentence)):
            pattern = 1
        elif(re11.match(sentence[::-1])):
            pattern = 2
    sentence = "{0:50}{1}{2:d}".format(sentence, '\t', pattern)
    f2.writelines(sentence + '\n')

f1.close()
f2.close()


