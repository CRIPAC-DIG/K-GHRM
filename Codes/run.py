import torch
import numpy as np
import os
import sys
from time import time
import warnings 
warnings.filterwarnings("ignore") 
from utility.batch_test import args, data_generator, pad_sequences, words_lookup, entity_lookup
from utility.models import *


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    config = dict()
    
    users_to_test = list(data_generator.test_set.keys())
    config['n_docs'] = data_generator.n_docs
    config['n_qrls'] = data_generator.n_qrls
    config['n_words'] = data_generator.n_words
    config['n_ents'] = data_generator.n_ents
    if args.model == 'KGIR':
        config['docs_adj'] = np.load("../Data/{}/doc_adj_{}.npy".format(args.dataset, str(args.doc_len)))
        config['ents_adj'] = np.load("../Data/{}/ent_adj_{}.npy".format(args.dataset, str(args.ent_num)))
    config['idf_dict'] = np.load("../Data/{}/idf.npy".format(args.dataset), allow_pickle=True).item()

    model = eval(args.model)(config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.l2)
    precision = []
    ndcg = []
    best_ret = 0
    best_epoch = 0
    for epoch in range(args.epoch):
        n_batch = 32
        total_loss = 0
        model.train()
        for idx in range(n_batch):
            qrls, pos_docs, neg_docs = data_generator.sample() # pos_docs and neg_docs contain the document ids

            pos_docs_words = pad_sequences(words_lookup(pos_docs), maxlen=args.doc_len, value=config['n_words'])
            neg_docs_words = pad_sequences(words_lookup(neg_docs), maxlen=args.doc_len, value=config['n_words'])
            pos_docs_ents = pad_sequences(entity_lookup(pos_docs), maxlen=args.ent_num, value=config['n_ents'])
            neg_docs_ents = pad_sequences(entity_lookup(neg_docs), maxlen=args.ent_num, value=config['n_ents'])

            l_word = [data_generator.qrl_word_list[i] for i in qrls]
            l_ent = []
            for i in qrls:
                if i in data_generator.qrl_ent_list:
                    l_ent.append(data_generator.qrl_ent_list[i])
                else:
                    l_ent.append([config['n_ents'] for _ in range(args.qrl_len)]) # pad those queries that do not have entities
            # l_ent = [data_generator.qrl_ent_list[i] for i in qrls]
            qrls_words = pad_sequences(l_word, maxlen=args.qrl_len, value=config['n_words'])
            qrls_ents = pad_sequences(l_ent, maxlen=args.qrl_len, value=config['n_ents'])

            pos_scores = model(qrls_words, pos_docs_words, qrls_ents, pos_docs_ents, pos_docs)
            neg_scores = model(qrls_words, neg_docs_words, qrls_ents, neg_docs_ents, neg_docs)
            loss = torch.max(torch.zeros_like(pos_scores).float().cuda(), (1 - pos_scores + neg_scores))
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.cpu().detach().numpy()
            optimizer.step()
        print("epoch:{}".format(epoch), "loss:{}".format(total_loss))
        
        ret = test(model, users_to_test)
        precision.append(ret[0])
        ndcg.append(ret[1])
        if ret[1] > best_ret:
            os.rename('./test.run', '{}.f{}.best.run'.format(args.dataset,str(args.fold)))
            best_ret = ret[1]
            best_epoch = epoch
    
    # precision = np.array(precision)
    # ndcg = np.array(ndcg)
    best_epoch = np.argmax(ndcg)

    print("best epoch:", best_epoch)
    print("P@20:", precision[best_epoch])
    print("ndcg@20:", ndcg[best_epoch])


    with open('results.txt', 'a') as f:
        f.writelines(str(args.fold) + '\t' + str(precision[best_epoch]) + '\t' + str(ndcg[best_epoch]) +'\n')

