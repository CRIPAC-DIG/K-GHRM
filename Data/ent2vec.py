from wikipedia2vec import Wikipedia2Vec
import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='robust04', help='dataset name: robust04/clueweb09')
args = parser.parse_args()

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

wiki2vec = Wikipedia2Vec.load('./enwiki_20180420_100d.pkl')
# wiki2vec = Wikipedia2Vec.load('./enwiki_20180420_300d.pkl')
ent2id = load_obj('./{}/ent2id'.format(args.dataset))

ent2vec = []
no_pretrain_emd_cnt = 0 
for e in ent2id:
    try:
        ent2vec.append(wiki2vec.get_entity_vector(e))
    except:
        no_pretrain_emd_cnt += 1
        ent2vec.append(np.random.randn(100))
        # ent2vec.append(np.random.randn(300))
print(no_pretrain_emd_cnt) # clueweb09:22820, robust04:8423
print(len(ent2vec)) # clueweb09:226363, robust04:108627
np.save('./{}/ent_embedding_100d.npy'.format(args.dataset), ent2vec)
# np.save('./{}/ent_embedding_300d.npy'.format(args.dataset), ent2vec)

# que_ent = load_obj('./{}/que_entity'.format(args.dataset))
# with open('./{}/que_entity_list_unique.txt'.format(args.dataset), 'w') as f:
#     for i in que_ent:
#         f.writelines(str(i)+'\t')
#         for word in que_ent[i]:
#             f.writelines(str(ent2id[word])+' ')
#         f.writelines('\n')
