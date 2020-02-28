import os
import json
import ahocorasick
from tqdm import tqdm

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
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root
        self.dataset = hyper.train
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r',encoding='utf-8'))

        self.reversed_relation_vocab = {
            v: k
            for k, v in self.relation_vocab.items()
        }

        # (s,p,o)三元组搜素
        self.s_ac = AC_Unicode()
        self.o_ac = AC_Unicode()
        self.so2p = {}
        self.sp2o = {}
        self.op2s = {}

        for line in open(os.path.join(self.data_root, self.dataset), 'r',encoding='utf-8'):
            line = line.strip("\n")
            instance = json.loads(line)
            spo_list = instance['spo_list']
            for spo in spo_list:
                s_ = ''.join(spo['subject'])
                o_ = ''.join(spo['object'])
                p = spo['predicate']
                self.s_ac.add_word(s_,(s_,spo['subject']))
                self.o_ac.add_word(o_,(o_,spo['object']))
                if (s_,o_) not in self.so2p:
                    self.so2p[(s_,o_)] = set()
                self.so2p[(s_,o_)].add(p)
        
        self.s_ac.make_automaton()
        self.o_ac.make_automaton()

    def extract_items(self, text):
        result = []
        for s in self.s_ac.iter(text):
            for o in self.o_ac.iter(text):
                if (s[1][0],o[1][0]) in self.so2p:
                    for p in self.so2p[(s[1][0],o[1][0])]:
                        triplet = {
                            'object':o[1][0],
                            'predicate':p,
                            'subject':s[1][0],
                            'score':0.0,
                            'distant': 1
                        }
                    if triplet not in result:
                        result.append(triplet)
        return result

    def extract_items_list(self, text):
        result = []
        for s in self.s_ac.iter(text):
            for o in self.o_ac.iter(text):
                if (s[1][0],o[1][0]) in self.so2p:
                    for p in self.so2p[(s[1][0],o[1][0])]:
                        triplet = {
                            'object':o[1][1],
                            'predicate':p,
                            'subject':s[1][1],
                            'score':0.0,
                            'distant': 1
                        }
                    if triplet not in result:
                        result.append(triplet)
        return result
    

                        

                

        