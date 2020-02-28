import os
import json
import pickle as pickle
import numpy as np
import random
import ahocorasick
from scipy.spatial import distance_matrix

"""
negative sampler
"""
class AC_Unicode:
    """
    AC automaton
    """
    def __init__(self):
        self.ac = ahocorasick.Automaton()
    def add_word(self,k,v):
        return self.ac.add_word(k,v)
    def make_automaton(self):
        return self.ac.make_automaton()
    def iter(self,s):
        return self.ac.iter(s)

class SPO_searcher:
    def __init__(self):
        #self.base_path = 'data/chinese_bert/multi_head_selection'
        self.base_path = 'data/chinese_bert/pso'
        self.relation_vocab = json.load(
            open(os.path.join(self.base_path, 'relation_vocab.json'), 'r',encoding='utf-8'))

        # (s,p,o)三元组搜素
        self.s_ac = AC_Unicode()
        self.o_ac = AC_Unicode()
        self.so2p = {}
        self.sp2o = {}
        self.op2s = {}

        for line in open(os.path.join(self.base_path, 'train_data.json'), 'r',encoding='utf-8'):
            line = line.strip("\n")
            instance = json.loads(line)
            spo_list = instance['spo_list']
            for spo in spo_list:
                s_ = ''.join(spo['subject'])
                o_ = ''.join(spo['object'])
                p = spo['predicate']
                self.s_ac.add_word(s_,s_)
                self.o_ac.add_word(o_,o_)
                if (s_,o_) not in self.so2p:
                    self.so2p[(s_,o_)] = set()
                self.so2p[(s_,o_)].add(p)
        
        self.s_ac.make_automaton()
        self.o_ac.make_automaton()

    def extract_items(self,text,spo_list):
        
        s_list,o_list = [''.join(spo['subject']) for spo in spo_list],[''.join(spo['object']) for spo in spo_list]
        s_o_list = set(zip(s_list,o_list))

        def _find_end_idx(q_list,k_list):
            # 在列表k种确定列表q的位置
            q_list_length = len(q_list)
            k_list_length = len(k_list)
            for idx in range(k_list_length - q_list_length + 1):
                t = [q == k for q,k in zip(q_list,k_list[idx:idx+q_list_length])]
                if all(t):
                    idx_start = idx
                    idx_end = idx + q_list_length - 1
                    return idx_start,idx_end
            return -1,-1

        res = []
        for s in self.s_ac.iter(text):
            for o in self.o_ac.iter(text):
                if (s[1],o[1]) in self.so2p and (s[1],o[1]) not in s_o_list:
                    s_s,s_e = _find_end_idx(s[1],text)
                    o_s,o_e = _find_end_idx(o[1],text)
                    if s_e != -1 and o_e != -1:
                        for p in self.so2p[(s[1],o[1])]:
                            triplet = {
                                'subject':s[1],
                                'predicate':p,
                                'object':o[1],
                                's_s': s_s+1,
                                's_e':s_e+1,
                                'o_s':o_s+1,
                                'o_e':o_e+1,
                                'p_idx':self.relation_vocab[p]
                            }
        
                            res.append(triplet)

        return res


class Negative_Sampler(object):
    def __init__(self):
        self.base_path = 'data/chinese_bert/multi_head_selection'
        #self.base_path = 'data/chinese_bert/pso'
        self.relation_vocab = json.load(
            open(os.path.join(self.base_path,'relation_vocab.json'), 'r',encoding='utf-8'))

        self.reversed_relation_vocab = {
            v: k for k, v in self.relation_vocab.items()
            }

        self.spo_maker = SPO_searcher()
        self.rel_emb = np.load("data/transe/train/rel_emb.npy")
        self.p_knn_dict = self.p_knn_map()

    def spo_search_res(self):
        with open(os.path.join('error/chinese_bert_pso', 'dev.json'), 'r',encoding='utf-8') as f, \
            open(os.path.join(self.base_path, 'dev_data_distillation.json'),'r',encoding = 'utf-8') as f1, \
            open(os.path.join('error/chinese_bert_pso','dev_new.json'),'w',encoding = 'utf-8') as t:
            for line1,line2 in zip(f,f1):
                line2= line2.strip("\n")
                instance = json.loads(line2)
                line1 = line1.strip("\n")
                neg = json.loads(line1)['neg']
                instance['neg'] = neg
                ins_json = json.dumps(instance, ensure_ascii=False)
                t.write(ins_json)
                t.write('\n')
                
        # drop old put new
        os.system("rm -rf error/chinese_bert_pso/dev.json")
        os.system("mv error/chinese_bert_pso/dev_new.json error/chinese_bert_pso/dev.json")

    def p_rnd(self):
        p_new_idx = random.randint(0,len(self.relation_vocab)-2)
        return p_new_idx

    def p_knn_map(self):
        dict_ = {}
        dis = distance_matrix(self.rel_emb,self.rel_emb)

        def find_top_k(dis,index,k=5):
            v = dis[index,:]
            v_index = np.argsort(v)
            return v_index[1:k+1]

        for i in range(len(self.rel_emb)):
            dict_[i] = list(find_top_k(dis,i))
        return dict_

    def p_knn_rnd(self):
        p_new_idx = random.randint(0,4)
        return p_new_idx

    def spo_rnd(self,res):
        #res = self.spo_maker.extract_items(text)
        len_ = len(res)
        if len_ == 0:
            return None

        new_idx = random.randint(0,len_-1)
        new_spo = res[new_idx]

        return new_spo


    def _find_end_idx(self,q_list,k_list):
        q_list_length = len(q_list)
        k_list_length = len(k_list)
        for idx in range(k_list_length - q_list_length + 1):
            t = [q == k for q,k in zip(q_list,k_list[idx:idx+q_list_length])]
            if all(t):
                idx_start = idx
                idx_end = idx + q_list_length - 1
                return idx_start,idx_end
        return -1,-1
    
    def run(self):

        def p_permutation():
            if random.random() <= 0.3:
                p_new_idx = self.p_rnd()
            else:
                p_new_idx =  self.p_knn_dict[p_idx][self.p_knn_rnd()]
            triplet = {
                        'subject':s_,
                        'predicate':self.reversed_relation_vocab[p_new_idx],
                        'object':o_,
                        's_s':s_s+1,
                        's_e':s_e+1,
                        'o_s':o_s+1,
                        'o_e':o_e+1,
                        'p_idx':int(p_new_idx)
                    }  
            return triplet   

        with open(os.path.join(self.base_path, 'train_data.json'), 'r',encoding='utf-8') as f, \
            open(os.path.join(self.base_path,'pos_neg.json'),'w',encoding = 'utf-8') as t:
            for line in f:
                line = line.strip("\n")
                instance = json.loads(line)
                spo_list = instance['spo_list']
                spo_res = self.spo_maker.extract_items(''.join(instance['text']),spo_list)
                pos,neg = [],[]
                if len(spo_list) == 1:
                    spo = spo_list[0]
                    s_ = ''.join(spo['subject'])
                    o_ = ''.join(spo['object'])
                    p = spo['predicate']
                    p_idx = self.relation_vocab[p]
                    s_s,s_e = self._find_end_idx(spo['subject'],instance['text'])
                    o_s,o_e = self._find_end_idx(spo['object'],instance['text'])
                    # distant first
                    tmp = self.spo_rnd(spo_res)
                    if tmp and random.random() <= 0.8:
                        triplet = tmp
                    else:
                        triplet = p_permutation()
                    neg.append(triplet)
                    pos.append(
                        {
                            'subject':s_,
                            'predicate':self.reversed_relation_vocab[p_idx],
                            'object':o_,
                            's_s':s_s+1,
                            's_e':s_e+1,
                            'o_s':o_s+1,
                            'o_e':o_e+1,
                            'p_idx':int(p_idx)
                        }
                    )
                else:
                    # more than once
                    for spo in spo_list:
                        s_ = ''.join(spo['subject'])
                        o_ = ''.join(spo['object'])
                        p = spo['predicate']
                        p_idx = self.relation_vocab[p]
                        s_s,s_e = self._find_end_idx(spo['subject'],instance['text'])
                        o_s,o_e = self._find_end_idx(spo['object'],instance['text'])
                        tmp = self.spo_rnd(spo_res)
                        pos.append(
                                {
                                    'subject':s_,
                                    'predicate':self.reversed_relation_vocab[p_idx],
                                    'object':o_,
                                    's_s':s_s+1,
                                    's_e':s_e+1,
                                    'o_s':o_s+1,
                                    'o_e':o_e+1,
                                    'p_idx':int(p_idx)
                                }  
                            )
                        if tmp and random.random() <= 0.7:
                            triplet = tmp
                        else:
                            triplet = p_permutation()
                        if triplet not in neg:
                            neg.append(triplet)
                    num_ = random.randint(1,len(neg))
                    rnd_idx = random.sample(range(len(neg)),num_)
                    neg = [neg[idx] for idx in rnd_idx]
                    

                # write into
                instance['pos'] = pos
                instance['neg'] = neg
                ins_json = json.dumps(instance, ensure_ascii=False)
                t.write(ins_json)
                t.write('\n')

tester = Negative_Sampler()
tester.run()
#tester.spo_search_res()







