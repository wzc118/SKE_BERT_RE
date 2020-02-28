import json
import codecs
from typing import Dict, List, Tuple, Set, Optional
from abc import ABC, abstractmethod
import os
import pandas as pd
#from overrides import overrides


class F1_abc(object):
    def __init__(self):
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10
        self.df = []
        self.idx = 0

    def reset(self) -> None:
        self.A = 1e-10
        self.B = 1e-10
        self.C = 1e-10
        self.df = []
        self.idx = 0

    def get_metric(self, reset: bool = False):
        if reset:
            self.reset()

        f1, p, r = 2 * self.A / (self.B +
                                 self.C), self.A / self.B, self.A / self.C
        result = {"precision": p, "recall": r, "fscore": f1}

        return result

    def __call__(self, predictions,
                 gold_labels):
        raise NotImplementedError

class F1_P(F1_abc):
    def __call__(self,predictions,golden_labels):
        for g,p in zip(golden_labels,predictions):
            g_set = set(gg for gg in g)
            p_set = set(pp for pp in p)
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)
            
            


class F1_triplet(F1_abc):

    #@overrides
    def __call__(self, predictions: List[List[Dict[str, str]]],
                 gold_labels: List[List[Dict[str, str]]]):

        for g, p in zip(gold_labels, predictions):
            try:
                g_set = set('_'.join((gg['object'], gg['predicate'],
                                    gg['subject'])) for gg in g)
                p_set = set('_'.join((pp['object'], pp['predicate'],
                                    pp['subject'])) for pp in p)
            except:
                g_set = set('_'.join((''.join(gg['object']), gg['predicate'],
                                    ''.join(gg['subject']))) for gg in g)
                p_set = set('_'.join((''.join(pp['object']), pp['predicate'],
                                    ''.join(pp['subject']))) for pp in p)

            format_ = {'num':self.idx,'set_l':p_set,'size_l':len(p_set),'set_r':g_set,'size_r':len(g_set),'match':g_set & p_set,'size_match':len(g_set & p_set)}
            self.A += len(g_set & p_set)
            self.B += len(p_set)
            self.C += len(g_set)
            self.df.append(format_)

    def get_df(self):
        df = pd.DataFrame.from_records(self.df)
        inter = sum(df["size_match"].values)
        left = sum(df["size_l"].values)
        right = sum(df["size_r"].values)

        print("dataframe result")
        f1, p, r = 2 * inter / (left + right), inter / left, inter / right
        result = {"precision": p, "recall": r, "fscore": f1}
        print('Triplets-> ' +  ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in result.items() if not name.startswith("_")]))

        df.to_csv('error/chinese_bert_re/post_check.csv')

class F1_ner(F1_abc):

    #@overrides
    def __call__(self, predictions: List[List[str]], gold_labels: List[List[str]]):
        for g, p in zip(gold_labels, predictions):

            inter = sum(tok_g == tok_p and tok_g in ('B', 'I')
                        for tok_g, tok_p in zip(g, p))
            bi_g = sum(tok_g in ('B', 'I') for tok_g in g)
            bi_p = sum(tok_p in ('B', 'I') for tok_p in p)

            self.A += inter
            self.B += bi_g
            self.C += bi_p

class SaveError():
    def __init__(self):
        self.p_error = dict()
        self.r_error = dict()
        self.comp_dict = []

    def save(self,batch,predictions,gold_labels):
        for i,(g,p) in enumerate(zip(gold_labels,predictions)):
            try:
                g_set = set('_'.join((gg['object'], gg['predicate'],
                                gg['subject'])) for gg in g)
                p_set = set('_'.join((pp['object'], pp['predicate'],
                                pp['subject'])) for pp in p)
            except:
                g_set = set('_'.join((''.join(gg['object']), gg['predicate'],
                                    ''.join(gg['subject']))) for gg in g)
                p_set = set('_'.join((''.join(pp['object']), pp['predicate'],
                                    ''.join(pp['subject']))) for pp in p)

            err_p = p_set - p_set&g_set
            err_r = g_set - p_set&g_set
            if len(err_p):
                self.p_error[str(batch) + '-' + str(i)] = list(err_p)
            if len(err_r):
                self.r_error[str(batch) + '-' + str(i)] = list(err_r)

            if p_set != g_set:
                self.comp_dict.append(
                    {
                        'num':str(batch) + '-' + str(i),
                        'predict': p,
                        'gold':g
                    }
                )

    def write(self,exp_name):
        # json dump
        if not os.path.exists(os.path.join('error',exp_name)):
            os.mkdir(os.path.join('error',exp_name))
        with codecs.open(os.path.join('error',exp_name,'p_error.json'),'w',encoding='utf=8') as f:
            json.dump([self.p_error],f,indent=4,ensure_ascii=False)
        
        with codecs.open(os.path.join('error',exp_name,'r_error.json'),'w',encoding='utf=8') as f:
            json.dump([self.r_error],f,indent=4,ensure_ascii=False)

        with codecs.open(os.path.join('error',exp_name, 'comp.json'),'w',encoding='utf=8') as f:
            json.dump(self.comp_dict,f,indent=4,ensure_ascii=False)      
        

class SaveRecord():
    def __init__(self,exp_name):
        self.exp_name = exp_name
        if not os.path.exists(os.path.join('error',exp_name)):
            os.mkdir(os.path.join('error',exp_name))

    def reset(self,file_name):
        self.f = open(os.path.join('error',self.exp_name, file_name),'w',encoding='utf=8')

    def save(self,neg_list,pos_list = None):
        dict_ = {}
        flag = False
        if pos_list:
            flag = True
        for i in range(len(neg_list)):
            dict_['neg'] = neg_list[i]
            if flag:
                dict_['pos'] = pos_list[i]
            neg_json = json.dumps(dict_, ensure_ascii=False)
            self.f.write(neg_json)
            self.f.write('\n')

"""
class SaveRecord():
    def __init__(self):
        self.p_rc = []

    def save(self,batch,neg_list,pos_list = None):
        for i in range(len(neg_list)):
            rc = {
                 'num':str(batch) + '-' + str(i),
                 'neg_list':neg_list[i]
            }
            if pos_list:
                rc['pos_list'] = pos_list
            self.p_rc.append(rc)

    def write(self,exp_name,file_name):
        if not os.path.exists(os.path.join('error',exp_name)):
            os.mkdir(os.path.join('error',exp_name))

        with codecs.open(os.path.join('error',exp_name, file_name),'w',encoding='utf=8') as f:
            json.dump(self.p_rc,f,ensure_ascii=False)      
"""
                        