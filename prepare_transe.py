import os
import json
from tqdm import tqdm
import pandas as pd
import argparse
"""
train for postcheck model: use train data
test for dev: use train data add dev record
"""

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    '-e',
                    type=str,
                    default='train',
                    help='train or dev')
args = parser.parse_args()

class TransE_Data(object):
    def __init__(self, is_train):
        self.is_train = True if is_train == 'train' else False
        #self.data_path = 'data/chinese_bert/multi_head_selection/train_data.json'
        self.data_path = 'data/chinese_bert/pso/train_data.json'
        self.r_vocab = json.load(
            open(os.path.join('data','chinese_bert','pso','relation_vocab.json'), 'r',encoding='utf-8'))

        if not self.is_train:
            self.dev_record_f = open(os.path.join('error','chinese_bert_pso','dev.json'),'r',encoding = 'utf-8')

        self.e_vocab = dict()
        self.spo_cnt = 0

        self.gen_vocab()
        if not self.is_train:
            self.gen_dev_vocab()

    def gen_vocab(self):
        idx = 0
        for line in open(self.data_path,'r',encoding = 'utf-8'):
            tmp = json.loads(line)['spo_list']
            for spo in tmp:
                s,p,o = ''.join(spo['subject']),spo['predicate'],''.join(spo['object'])
                s,o = ''.join(s.split(' ')),''.join(o.split(' '))
                self.spo_cnt += 1
                if s not in self.e_vocab:
                    self.e_vocab[s] = idx
                    idx += 1
                if o not in self.e_vocab:
                    self.e_vocab[o] = idx
                    idx += 1
        self.index = idx


    def gen_entity2id(self):
        if self.is_train:
            if not os.path.exists(os.path.join('data','transe','train')):
                os.makedirs(os.path.join('data','transe','train'))
        else:
            if not os.path.exists(os.path.join('data','transe','dev')):
                os.makedirs(os.path.join('data','transe','dev'))

        path = os.path.join('data','transe','train','entity2id_pso.txt') if self.is_train else \
            os.path.join('data','transe','dev','entity2id_pso.txt') 

        with open(path,'w',encoding = 'utf-8') as f:
            f.write(str(len(self.e_vocab)))
            f.write('\n')
            for k,v in self.e_vocab.items():
                f.write(k + '\t' + str(v) + '\n')
        f.close()
    
    def gen_relation2id(self):

        path = os.path.join('data','transe','train','relation2id_pso.txt') if self.is_train else \
            os.path.join('data','transe','dev','relation2id_pso.txt')         

        del self.r_vocab['N']

        with open(path,'w',encoding = 'utf-8') as f:
            f.write(str(len(self.r_vocab)))
            f.write('\n')
            for k,v in self.r_vocab.items():
                f.write(k + '\t' + str(v) + '\n')
        f.close()        

    def gen_train2id(self):

        path = os.path.join('data','transe','train','train2id_pso.txt') if self.is_train else \
            os.path.join('data','transe','dev','train2id_pso.txt')   

        with open(path,'w',encoding = 'utf-8') as f:
            f.write(str(self.spo_cnt))
            f.write('\n')
            for line in open(self.data_path,'r',encoding = 'utf-8'):
                tmp = json.loads(line)['spo_list']
                for spo in tmp:
                    s,p,o = ''.join(spo['subject']),spo['predicate'],''.join(spo['object'])
                    s,o = ''.join(s.split(' ')),''.join(o.split(' '))
                    s_idx,p_idx,o_idx = self.e_vocab[s],self.r_vocab[p],self.e_vocab[o]
                    f.write(str(s_idx) + '\t' + str(o_idx) + '\t' + str(p_idx) + '\n')
        f.close()

    def gen_dev_vocab(self):
        idx = self.index
        for line in self.dev_record_f:
            tmp = json.loads(line)['neg']
            for spo in tmp:
                s,p,o = spo['subject'],spo['predicate'],spo['object']
                s,o = ''.join(s.split(' ')),''.join(o.split(' '))
                self.spo_cnt += 1
                if s not in self.e_vocab:
                    self.e_vocab[s] = idx
                    idx += 1
                if o not in self.e_vocab:
                    self.e_vocab[o] = idx
                    idx += 1
        
    def gen_dev_train2id(self):
        path = os.path.join('data','transe','train','train2id.txt') if self.is_train else \
            os.path.join('data','transe','dev','train2id.txt')

        self.dev_record_f = open(os.path.join('error','chinese_bert_pso','dev.json'),'r',encoding = 'utf-8')

        with open(path,'a',encoding = 'utf-8') as t:
            for line in self.dev_record_f:
                tmp = json.loads(line)['neg']
                for spo in tmp:
                    s,p,o = spo['subject'],spo['predicate'],spo['object']
                    s,o = ''.join(s.split(' ')),''.join(o.split(' '))
                    s_idx,p_idx,o_idx = self.e_vocab[s],self.r_vocab[p],self.e_vocab[o]
                    t.write(str(s_idx) + '\t' + str(o_idx) + '\t' + str(p_idx) + '\n')
        t.close()

if __name__ == "__main__":
    test = TransE_Data(args.mode)
    test.gen_entity2id()
    test.gen_relation2id()
    test.gen_train2id()
    if args.mode == 'dev':
        test.gen_dev_train2id()
