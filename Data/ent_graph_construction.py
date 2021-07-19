import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import os
import warnings 
from tqdm import tqdm, trange
import multiprocessing
import argparse
import pickle

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

warnings.filterwarnings("ignore") 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='robust04', help='dataset name: robust04/clueweb09')
parser.add_argument('--window_size', default=5, help='window size')
parser.add_argument('--ent_pad_len', default=50, help='document pad length')
args = parser.parse_args()
path = './{}/'.format(args.dataset)

doc_file = path + 'doc_entity'
que_file = path + 'que_entity'
id_file = path + 'ent2id'
doc_ent = load_obj(doc_file)
que_ent = load_obj(que_file)
ent2id = load_obj(id_file)

unique_words = path + 'doc_ent_list_unique.txt'
unique_que_ents = path + 'que_ent_list_unique.txt'
n_docs = 0
n_words = 0
doc_word_list = []
doc_ent_list_unique = []
ENT_PAD_LEN = int(args.ent_pad_len)

window_size = args.window_size
windows = []
n_docs = len(doc_ent)
n_ques = len(que_ent)

for ent_list in doc_ent.values():
    words = [ent2id[e] for e in ent_list]
    length = len(words)
    doc_word_list.append(words)
    # sliding windows
    window = []
    for j in range(length - window_size + 1):
        window += [words[j: j + window_size]]
    windows.append(window)
    n_words = max(n_words, max(words))

#n_docs += 1
n_words += 1

def pad_sequences(items, maxlen, value=n_words):
    result = []
    for item in items:
        if len(item) < maxlen:
            item = item + [value] * (maxlen - len(item))
        if len(item) > maxlen:
            item = item[:maxlen]
        result.append(item)
    return result
    
def normalized_adj_bi(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
'''
# query unique entities
que_ent_list = []
que_ent_list_unique = []
for ent_list in que_ent.values():
    words = [ent2id[e] for e in ent_list]
    que_ent_list.append(words)

for queid in range(n_ques):
    que_ent_unique = set(que_ent_list[queid])
    que_ent_list_unique.append(sorted(que_ent_unique, key=que_ent_list[queid].index))

with open(unique_que_ents, 'w') as f:
    for i in range(len(que_ent_list_unique)):
        f.writelines(str(i)+'\t')
        for word in que_ent_list_unique[i]:
            f.writelines(str(word)+' ')
        f.writelines('\n')
'''

for docid in range(n_docs):
    # print(docid)
    word_list_unique = set(doc_word_list[docid])
    doc_ent_list_unique.append(sorted(word_list_unique, key=doc_word_list[docid].index))
'''
with open(unique_words, 'w') as f:
    for i in range(len(doc_ent_list_unique)):
        f.writelines(str(i)+'\t')
        for word in doc_ent_list_unique[i]:
            f.writelines(str(word)+' ')
        f.writelines('\n')
'''
padded_words = pad_sequences(doc_ent_list_unique, ENT_PAD_LEN)
def func(start, end):
    if end > n_docs: 
        end = n_docs
    batch_adj = []
    for k in trange(start, end):
        R = sp.dok_matrix((n_words+1, n_words+1), dtype=np.float32) 
        for window in windows[k]:
            for i in range(1, len(window)):
                for j in range(0, i):
                    word_i = window[i]
                    word_j = window[j]
                    if word_i == word_j:
                        continue
                    R[word_i,word_j] += 1.
                    R[word_j,word_i] += 1.
        R = R.tocsc()
        adj = R[padded_words[k],:][:,padded_words[k]].toarray()
        batch_adj.append(normalized_adj_bi(adj))
    return np.stack(batch_adj, 0)

# # multiprocessing for large collections:
pool = multiprocessing.Pool(processes=20)
res = []
t = n_docs // 20
for i in range(21):
    res.append(pool.apply_async(func, (i*t,(i+1)*t)))
pool.close()
pool.join()

r = [i.get() for i in res]
arr = np.concatenate(r, 0)
np.save(path + "/ent_adj_{}.npy".format(args.ent_pad_len),arr)

# np.save(path + "/doc_adj.npy",func(0, n_docs))