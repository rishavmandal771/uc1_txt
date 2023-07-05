import os
from pytorch_transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import pickle
import random
import numpy as np
import collections
from collections import Counter
import sys


def is_english_char(cp):
    """Checks whether CP is the codepoint of an English character."""
    if ((cp >= 0x0041 and cp <= 0x005A) or  # uppercase A-Z
        (cp >= 0x0061 and cp <= 0x007A) or  # lowercase a-z
        (cp >= 0x00C0 and cp <= 0x00FF) or  # Latin-1 Supplement
        (cp >= 0x0100 and cp <= 0x017F) or  # Latin Extended-A
        (cp >= 0x0180 and cp <= 0x024F) or  # Latin Extended-B
        (cp >= 0x1E00 and cp <= 0x1EFF) or  # Latin Extended Additional
        (cp >= 0x2C60 and cp <= 0x2C7F) or  # Latin Extended-C
        (cp >= 0xA720 and cp <= 0xA7FF) or  # Latin Extended-D
        (cp >= 0xAB30 and cp <= 0xAB6F) or  # Latin Extended-E
        (cp >= 0xFB00 and cp <= 0xFB06)):  # Alphabetic Presentation Forms
        return True

    return False


def nobert4token(tokenizer, title, attribute, value):

    def get_char(sent):
        tmp = []
        s = ''
        for char in sent.strip():
            if char.strip():
                cp = ord(char)
                if is_english_char(cp):
                    if s:
                        tmp.append(s)
                    tmp.append(char)
                    s = ''
                else:
                    s += char
            elif s:
                tmp.append(s)
                s = ''
        if s:
            tmp.append(s)
        return tmp

    title_list = get_char(title)
    attribute_list = get_char(attribute)
    value_list = get_char(value)

    tag_list = ['O']*len(title_list)
    for i in range(0,len(title_list)-len(value_list)):
        if title_list[i:i+len(value_list)] == value_list:
            for j in range(len(value_list)):
                if j==0:
                    tag_list[i+j] = 'B'
                else:
                    tag_list[i+j] = 'I'

    title_list = tokenizer.convert_tokens_to_ids(title_list)
    attribute_list = tokenizer.convert_tokens_to_ids(attribute_list)
    value_list = tokenizer.convert_tokens_to_ids(value_list)
    tag_list = [TAGS[i] for i in tag_list]

    return title_list, attribute_list, value_list, tag_list


max_len =100
def X_padding(ids):
    if len(ids) >= max_len:  
        return ids[:max_len]
    ids.extend([0]*(max_len-len(ids))) 
    return ids

tag_max_len = 6
def tag_padding(ids):
    if len(ids) >= tag_max_len: 
        return ids[:tag_max_len]
    ids.extend([0]*(tag_max_len-len(ids))) 
    return ids

    if len(ids) >= tag_max_len: 
        return ids[:tag_max_len]
    ids.extend([0]*(tag_max_len-len(ids))) 
    return ids

def rawdata2pkl4nobert(path):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    titles = []
    attributes = []
    values = []
    tags = []
    with open(path, 'r',encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]

        for index, line in enumerate(tqdm(lines[:100000])):
                title, attribute, value = line.split('$$$')
                title, attribute, value, tag = nobert4token(tokenizer, title, attribute, value)
                titles.append(title)
                attributes.append(attribute)
                values.append(value)
                tags.append(tag)

    print([tokenizer.convert_ids_to_tokens(i) for i in titles[:1]])
    print([[id2tags[j] for j in i] for i in tags[:1]])
    print([tokenizer.convert_ids_to_tokens(i) for i in attributes[:1]])
    print([tokenizer.convert_ids_to_tokens(i) for i in values[:1]])

    df = pd.DataFrame({'titles': titles, 'attributes': attributes, 'values': values, 'tags': tags},
                      index=range(len(titles)))
    print(df.shape)
    df['x'] = df['titles'].apply(X_padding)
    df['y'] = df['tags'].apply(X_padding)
    df['att'] = df['attributes'].apply(tag_padding)
    df['val']=df['values'].apply(tag_padding)
    
    

    index = list(range(len(titles)))
    random.shuffle(index)
    train_index = index[:int(0.9 * len(index))]
    valid_index = index[int(0.9 * len(index)):int(0.96 * len(index))]
    test_index = index[int(0.96 * len(index)):]

    train = df.loc[train_index, :]
    valid = df.loc[valid_index, :]
    test = df.loc[test_index, :]
    print(train['x'].values)

    train_x = np.asarray(list(train['x'].values))
    train_att = np.asarray(list(train['att'].values))
    train_y = np.asarray(list(train['y'].values))

    valid_x = np.asarray(list(valid['x'].values))
    valid_att = np.asarray(list(valid['att'].values))
    valid_y = np.asarray(list(valid['y'].values))
    

    test_x = np.asarray(list(test['x'].values))
    test_att = np.asarray(list(test['att'].values))
    test_value = np.asarray(list(test['val'].values))
    test_y = np.asarray(list(test['y'].values))
    
   
    print(train_x[0:1])
    with open('../data/container.pkl', 'wb') as outp:
        pickle.dump(train_x, outp)
        pickle.dump(train_att, outp)
        pickle.dump(train_y, outp)
        pickle.dump(valid_x, outp)
        pickle.dump(valid_att, outp)
        pickle.dump(valid_y, outp)
        pickle.dump(test_x, outp)
        pickle.dump(test_att, outp)
        pickle.dump(test_value, outp)
        pickle.dump(test_y, outp)
if __name__=='__main__':
    TAGS = {'':0,'B':1,'I':2,'O':3}
    id2tags = {v:k for k,v in TAGS.items()}
    path = '../data/output_ps.txt'
    #rawdata2pkl4bert(path, att_list)
    rawdata2pkl4nobert(path)