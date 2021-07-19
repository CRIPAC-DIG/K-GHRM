import pickle
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

doc_entity = load_obj('./{}/doc_entity'.format(args.dataset))
que_entity = load_obj('./{}/que_entity'.format(args.dataset))


ent2id = {}
ent2vec = []
id_ent = 0 # start numbering from zero
for e_list in que_entity.values():
    for e in e_list:
        if e not in ent2id:
            ent2id[e] = id_ent
            id_ent += 1

for e_list in doc_entity.values():
    for e in e_list:
        if e not in ent2id:
            ent2id[e] = id_ent
            id_ent += 1

save_obj(ent2id, './{}/ent2id'.format(args.dataset))


# print(len(load_obj('ent2id')))  there are 109462 unique entities in all queries and documents