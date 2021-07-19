import tagme
import multiprocessing
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec
import pickle
import argparse

# Set the authorization token for subsequent calls.
tagme.GCUBE_TOKEN = "4a8f6824-e873-4380-a079-cc859e5fe4de-843339462"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='robust04', help='dataset name: robust04/clueweb09')
args = parser.parse_args()

with open('./{}/clean.documents.txt'.format(args.dataset)) as f:
    doc_lines = f.readlines()
with open('./{}/clean.queries.txt'.format(args.dataset)) as f:
    que_lines = f.readlines()
wiki2vec = Wikipedia2Vec.load('./enwiki_20180420_100d.pkl')

def func(start, end):
    word2ent = {}
    cnt = start
    for line in tqdm(doc_lines[start:end]):
        sent = line.split('\t')[1].strip()
        lunch_annotations = tagme.annotate(sent)
        if lunch_annotations:
            for ann in lunch_annotations.get_annotations(0.1):
                if cnt not in word2ent:
                    try:
                        wiki2vec.get_entity_vector(ann.entity_title)
                        word2ent[cnt] = list(set([ann.entity_title]))
                    except:
                        print('no pretrained embedding for {}'.format(ann.entity_title))
                        continue
                else:
                    try:
                        wiki2vec.get_entity_vector(ann.entity_title)
                        word2ent[cnt] += list(set([ann.entity_title]))
                    except:
                        print('no pretrained embedding for {}'.format(ann.entity_title))
                        continue
            if cnt not in word2ent:
                print('no entities detected in {}'.format(sent))
                word2ent[cnt] = []
            cnt += 1
    return word2ent

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
'''
# extract entities in queries
word2ent = {}
cnt = 0
for line in tqdm(que_lines):
    sent = line.split('\t')[1].strip()
    lunch_annotations = tagme.annotate(sent)
    for ann in lunch_annotations.get_annotations(0.1):
        if cnt not in word2ent:
            try:
                wiki2vec.get_entity_vector(ann.entity_title)
                word2ent[cnt] = list(set([ann.entity_title]))
            except:
                print('no pretrained embedding for {}'.format(ann.entity_title))
                continue
        else:
            try:
                wiki2vec.get_entity_vector(ann.entity_title)
                word2ent[cnt] += list(set([ann.entity_title]))
            except:
                print('no pretrained embedding for {}'.format(ann.entity_title))
                continue
    if cnt not in word2ent:
        print('no entities detected in {}'.format(sent))
        word2ent[cnt] = []
    cnt += 1
save_obj(word2ent, './{}/que_entity'.format(args.dataset))
'''
# extract entities in documents
doc_entity = {}
pool = multiprocessing.Pool(processes=40)
res = []
t = len(doc_lines) // 40
for i in range(41):
    res.append(pool.apply_async(func, (i*t,(i+1)*t)))
pool.close()
pool.join()
for r in res:
    doc_entity.update(r.get())
save_obj(doc_entity, './{}/doc_entity'.format(args.dataset))