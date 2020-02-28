import os
import openke
from openke.module.model import TransE
from openke.data import TrainDataLoader
import numpy as np
import pickle

phase = 'train'
base_root = './benchmarks/transe_ske_pso' + '/' + phase + '/'
relation2id_path = os.path.join(base_root,'relation2id.txt')
entity2id_path = os.path.join(base_root,'entity2id.txt')

train_dataloader = TrainDataLoader(
    #in_path = "./benchmarks/transe_ske/", 
    in_path = base_root,
    nbatches = 100,
    threads = 8, 
    sampling_mode = "normal", 
    bern_flag = 1, 
    filter_flag = 1, 
    neg_ent = 25,
    neg_rel = 5)

# define the model
transe = TransE(
    ent_tot = train_dataloader.get_ent_tot(),
    rel_tot = train_dataloader.get_rel_tot(),
    dim = 200, 
    p_norm = 2, 
    norm_flag = True)

save_path = os.path.join('checkpoint',phase,'transe.ckpt')
transe.load_checkpoint(save_path)
rel_emb = transe.get_parameters()['rel_embeddings.weight']
ent_emb = transe.get_parameters()['ent_embeddings.weight']

e_emb, r_emb = dict(),dict()
with open(entity2id_path,'r',encoding = 'utf-8') as f:
    next(f)
    for line in f:
        tmp = line.split('\t')
        entity = ''.join(tmp[:-1])
        e_emb[entity] = ent_emb[int(tmp[1]),:]

with open(relation2id_path,'r',encoding = 'utf-8') as f:
    next(f)
    for line in f:
        tmp = line.split('\t')
        r_emb[tmp[0]] = rel_emb[int(tmp[1]),:]

with open('{}_ent_emb.pickle'.format(phase),'wb') as handle:
    pickle.dump(e_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('{}_rel_emb.pickle'.format(phase),'wb') as handle:
    pickle.dump(r_emb, handle, protocol=pickle.HIGHEST_PROTOCOL)

#np.save('rel_emb.npy',rel_emb)