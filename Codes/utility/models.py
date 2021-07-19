import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utility.parser import parse_args
from utility.batch_test import data_generator, test
args = parse_args()

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_docs = config['n_docs']
        self.n_qrls = config['n_qrls']
        self.n_words = config['n_words']
        self.n_ents = config['n_ents']
        # word embedding initialization
        pretrained_word_emd = np.load("../Data/{}/word_embedding_300d.npy".format(args.dataset))
        l2_norm = np.sqrt((pretrained_word_emd * pretrained_word_emd).sum(axis=1))
        pretrained_word_emd = pretrained_word_emd / l2_norm[:, np.newaxis]
        pretrained_word_emd = np.concatenate([pretrained_word_emd, np.zeros([1, pretrained_word_emd.shape[-1]])], 0) # add padding row
        self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_word_emd),freeze=True,padding_idx=self.n_words).cuda()
        self.word_embedding = self.word_embedding.float()
        # entity embedding initialization
        pretrained_ent_emd = np.load("../Data/{}/ent_embedding_100d.npy".format(args.dataset))
        l2_norm = np.sqrt((pretrained_ent_emd * pretrained_ent_emd).sum(axis=1))
        pretrained_ent_emd = pretrained_ent_emd / l2_norm[:, np.newaxis]
        pretrained_ent_emd = np.concatenate([pretrained_ent_emd, np.zeros([1, pretrained_ent_emd.shape[-1]])], 0)
        self.ent_embedding = nn.Embedding.from_pretrained(torch.from_numpy(pretrained_ent_emd),freeze=True,padding_idx=self.n_ents).cuda()
        self.ent_embedding = self.ent_embedding.float()

        self._init_weights()
    def _init_weights(self):
        raise NotImplementedError 
    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)
    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
    def create_simmat(self, a_emb, b_emb):
        BAT, A, B = a_emb.shape[0], a_emb.shape[1], b_emb.shape[1]
        a_denom = a_emb.norm(p=2, dim=2).reshape(BAT, A, 1).expand(BAT, A, B) + 1e-9 # avoid 0div
        b_denom = b_emb.norm(p=2, dim=2).reshape(BAT, 1, B).expand(BAT, A, B) + 1e-9 # avoid 0div
        perm = b_emb.permute(0, 2, 1)
        sim = a_emb.bmm(perm)
        sim = sim / (a_denom * b_denom)
        return sim

class KGIR(BaseModel):
    def _init_weights(self):
        self.word_adj = self.config['docs_adj']
        self.ent_adj = self.config['ents_adj']
        self.idf_dict = self.config['idf_dict']
        self.linear1 = nn.Linear(args.word_topk*3, 64).cuda()
        self.linear2 = nn.Linear(64, 32).cuda()
        self.linear3 = nn.Linear(32, 1).cuda()
        self.linear4 = nn.Linear(args.ent_topk*3, 32).cuda()
        self.linear5 = nn.Linear(32, 1).cuda()

        self.linearz0 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearz1 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearr0 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearr1 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearh0 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearh1 = nn.Linear(args.qrl_len, args.qrl_len).cuda()

        self.linearz02 = nn.Linear(1,1).cuda()
        self.linearz12 = nn.Linear(1,1).cuda()
        self.linearr02 = nn.Linear(1,1).cuda()
        self.linearr12 = nn.Linear(1,1).cuda()
        self.linearh02 = nn.Linear(1,1).cuda()
        self.linearh12 = nn.Linear(1,1).cuda()

        self.linearz03 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearz13 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearr03 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearr13 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearh03 = nn.Linear(args.qrl_len, args.qrl_len).cuda()
        self.linearh13 = nn.Linear(args.qrl_len, args.qrl_len).cuda()

        self.linearz04 = nn.Linear(1,1).cuda()
        self.linearz14 = nn.Linear(1,1).cuda()
        self.linearr04 = nn.Linear(1,1).cuda()
        self.linearr14 = nn.Linear(1,1).cuda()
        self.linearh04 = nn.Linear(1,1).cuda()
        self.linearh14 = nn.Linear(1,1).cuda()

        self.gated = nn.Linear(1, 1).cuda()
        self.linearp1 = nn.Linear(4,1).cuda()
        self.linearp2 = nn.Linear(4,1).cuda()
        # aggregation
        self.agg = nn.Linear(8,1).cuda()
        #self.dropout = nn.Dropout(args.dp)
    def ggnn1(self,feat,doc_ids,graph):
        if graph == 'word':
            adj = self.word_adj[doc_ids]
            adj = torch.FloatTensor(adj).cuda()
        elif graph == 'entity':
            adj = self.ent_adj[doc_ids]
            adj = torch.FloatTensor(adj).cuda()
        x = feat
        a = adj.matmul(x)

        z0 = self.linearz0(a)
        z1 = self.linearz1(x)
        z = F.sigmoid(z0 + z1)

        r0 = self.linearr0(a)
        r1 = self.linearr1(x)
        r = F.sigmoid(r0 + r1)

        h0 = self.linearh0(a)
        h1 = self.linearh1(r*x)
        h = F.relu(h0 + h1)

        feat = h*z + x*(1-z)
        #x = self.dropout(feat)
        return feat
    def ggnn2(self,feat,doc_ids,graph,keep_rate=0.8):
        if graph == 'word':
            adj = self.word_adj[doc_ids]
            adj = torch.FloatTensor(adj).cuda()
        elif graph == 'entity':
            adj = self.ent_adj[doc_ids]
            adj = torch.FloatTensor(adj).cuda()
        node_num = adj.shape[1]
        x = self.linearp1(feat)
        a = adj.matmul(x)

        z0 = self.linearz02(a)
        z1 = self.linearz12(x)
        z = F.sigmoid(z0 + z1)
        
        r0 = self.linearr02(a)
        r1 = self.linearr12(x)
        r = F.sigmoid(r0 + r1)

        h0 = self.linearh02(a)
        h1 = self.linearh12(r*x)

        h = F.relu(h0 + h1)
        
        feat_s = h*z + x*(1-z)

        score, indices = feat_s.topk(int(node_num*keep_rate),1)
        indices = torch.squeeze(indices)
        seq_feat = []
        seq_adj = []
        for i in range(feat.shape[0]):
            seq_feat.append(feat[i][indices[i]])
            seq_adj.append(adj[i,indices[i],:][:,indices[i]])
        feat = torch.stack(tuple(seq_feat))
        adj = torch.stack(tuple(seq_adj))
        feat = F.tanh(score) * feat
        return feat, adj
    
    def ggnn3(self,feat,adj):
        x = feat
        a = adj.matmul(x)
        
        z0 = self.linearz03(a)
        z1 = self.linearz13(x)
        z = F.sigmoid(z0 + z1)

        r0 = self.linearr03(a)
        r1 = self.linearr13(x)
        r = F.sigmoid(r0 + r1)

        h0 = self.linearh03(a)
        h1 = self.linearh13(r*x)

        h = F.relu(h0 + h1)
        
        feat = h*z + x*(1-z)
        return feat

    def ggnn4(self,feat,adj,keep_rate=0.8):
        x = self.linearp2(feat)
        a = adj.matmul(x)
        node_num = adj.shape[1]
        
        z0 = self.linearz04(a)
        z1 = self.linearz14(x)
        z = F.sigmoid(z0 + z1)
        
        r0 = self.linearr04(a)
        r1 = self.linearr14(x)
        r = F.sigmoid(r0 + r1)

        h0 = self.linearh04(a)
        h1 = self.linearh14(r*x)

        h = F.relu(h0 + h1)
        
        feat_s = h*z + x*(1-z)
        score, indices = feat_s.topk(int(node_num*keep_rate),1)
        indices = torch.squeeze(indices)
        seq_feat = []
        for i in range(feat.shape[0]):
            seq_feat.append(feat[i][indices[i]])
        feat = torch.stack(tuple(seq_feat))
        feat = F.tanh(score) * feat
        return feat
        
    def forward(self, qrl_token, doc_token, qrls_ents, docs_ents, doc_ids, test=False):
        self.test = test
        self.batch_size = len(qrl_token)
        self.idf = torch.FloatTensor([[self.idf_dict[word] for word in words] for words in qrl_token]).cuda().unsqueeze(-1)
        qrl_word_embedding = self.word_embedding(torch.tensor(qrl_token).long().cuda())
        doc_word_embedding = self.word_embedding(torch.tensor(doc_token).long().cuda())
        qrl_ent_embedding = self.ent_embedding(torch.tensor(qrls_ents).long().cuda())
        doc_ent_embedding = self.ent_embedding(torch.tensor(docs_ents).long().cuda())
        word_sim = self.create_simmat(qrl_word_embedding, doc_word_embedding).permute(0, 2, 1) #batch, len_d, len_q
        ent_sim = self.create_simmat(qrl_ent_embedding, doc_ent_embedding).permute(0, 2, 1)
        
        word_sim_per = word_sim.permute(0,2,1)
        topk_0, _ = word_sim_per.topk(args.word_topk,-1)

        rep1 = self.ggnn1(word_sim, doc_ids, 'word')
        att_x1, adj_new = self.ggnn2(rep1, doc_ids, 'word')

        rep3 = self.ggnn3(att_x1, adj_new)
        att_x2 = self.ggnn4(rep3, adj_new)

        #1-hop representation
        att_x1 = att_x1.permute(0,2,1)  #batch, qrl, doc
        att_x1, _ = att_x1.topk(args.word_topk,-1)
        #2-hop representation
        att_x2 = att_x2.permute(0,2,1) 
        att_x2, _ = att_x2.topk(args.word_topk,-1) 

        att_x = torch.cat((topk_0,att_x1,att_x2),dim=-1)
        rel = F.relu(self.linear1(att_x))
        rel = F.relu(self.linear2(rel))
        rel_word = self.linear3(rel) # shape: [4,1]
        if args.idf:
            gated_weight = F.softmax(self.gated(self.idf), dim=1)
            rel_word = rel_word * gated_weight

        # entity embedding propagation
        ent_sim_per = ent_sim.permute(0,2,1)
        topk_0, _ = ent_sim_per.topk(args.ent_topk,-1)

        rep1 = self.ggnn1(ent_sim, doc_ids, 'entity')
        att_x1, adj_new = self.ggnn2(rep1, doc_ids, 'entity')

        rep3 = self.ggnn3(att_x1,adj_new)
        att_x2 = self.ggnn4(rep3,adj_new)

        #1-hop representation
        att_x1 = att_x1.permute(0,2,1) 
        att_x1, _ = att_x1.topk(args.ent_topk,-1)
        #2-hop representation
        att_x2 = att_x2.permute(0,2,1) 
        att_x2, _ = att_x2.topk(args.ent_topk,-1) 

        att_x = torch.cat((topk_0,att_x1,att_x2),dim=-1)
        rel = F.relu(self.linear4(att_x))
        rel_ent = self.linear5(rel) # batch_size, qrl_len, 1
        # rel = rel_word + rel_ent
        # scores = rel.squeeze(-1).sum(-1, keepdim=True)
        scores = self.agg(torch.cat((rel_word, rel_ent),dim=1).permute(0,2,1))
        scores = scores.squeeze(-1)
        if test:
            scores = scores.reshape((1, -1))
        return scores