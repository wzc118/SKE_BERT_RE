import os
import json
from collections import Counter
from typing import Dict, List, Tuple, Set, Optional
import re
import itertools

from cached_property import cached_property
from transformers import *

class Simplebert(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("./bert-base-chinese")

    def _tokenize_(self,text):
        split_tokens = []
        flag = False
        if self.tokenizer.do_basic_tokenize:
            for token in self.tokenizer.basic_tokenizer.tokenize(text,never_split=self.tokenizer.all_special_tokens):
                for sign in ('~','～','cm','《','》',"─"):
                    if token.find(sign):
                        flag = True
                split_tokens.append(token)

        if flag:
            split_tokens = [word for line in split_tokens for word in list(filter(None,re.split(r'(cm|~|～|《|》|≫|─)',line)))]
        return split_tokens
    
class Chinese_bert_pso_preprocessing(object):
    def __init__(self, hyper):
        self.hyper = hyper
        self.raw_data_root = hyper.raw_data_root
        self.data_root = hyper.data_root
        self.schema_path = os.path.join(self.raw_data_root, 'all_50_schemas')
        self.tokener = Simplebert()
        self.word_vocab = Counter()

        if not os.path.exists(self.schema_path):
            raise FileNotFoundError(
                'schema file not found, please check your downloaded data!')
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        
        self.relation_vocab_path = os.path.join(self.data_root,
                                                hyper.relation_vocab)

        self.pso_vocab_path = os.path.join(self.data_root,'pso_schema.json')

    @cached_property
    def relation_vocab(self):
        if os.path.exists(self.relation_vocab_path):
            pass
        else:
            self.gen_relation_vocab()
        return json.load(open(self.relation_vocab_path, 'r',encoding='utf-8'))
    
    def gen_bio_vocab(self):
        result = {'<pad>': 3, 'B': 0, 'I': 1, 'O': 2}
        json.dump(result,
                  open(os.path.join(self.data_root, 'bio_vocab.json'), 'w',encoding='utf-8'))

    def gen_tag_vocab(self):
        result = {'[CLS]':0,'[SEP]':1,'B-SUB':2,'I-SUB':3,'B-OBJ':4,'I-OBJ':5,'O':6,'[Category]':7}
        json.dump(result,
                    open(os.path.join(self.data_root, 'tag_bio_vocab.json'),'w',encoding = 'utf-8'))

    def gen_relation_vocab(self):
        relation_vocab = {}
        i = 0
        for line in open(self.schema_path, 'r',encoding='utf-8'):
            relation = json.loads(line)['predicate']
            if relation not in relation_vocab:
                relation_vocab[relation] = i
                i += 1
        relation_vocab['N'] = i
        json.dump(relation_vocab,
                  open(self.relation_vocab_path, 'w',encoding = 'utf-8'),
                  ensure_ascii=False)

    def gen_pso_vocab(self):
        pso_vocab = {}
        for line in open(self.schema_path,'r',encoding = 'utf-8'):
            tmp = json.loads(line)
            predicate = tmp['predicate']
            subject_type,object_type = tmp['subject_type'],tmp['object_type']
            if object_type == "Number":
                object_type = "数字"
            elif object_type == "Text":
                object_type = "文字"
            elif object_type == "Date":
                object_type = "日期"
            pso_vocab.setdefault(predicate,[]).extend([subject_type,object_type])
        json.dump(pso_vocab,
                 open(self.pso_vocab_path,'w',encoding = 'utf-8'),
                 ensure_ascii=False)    

    def gen_vocab(self, min_freq: int):
        target = os.path.join(self.data_root, 'word_vocab.json')
        result = {'<pad>':0}
        i = 1
        for k,v in self.word_vocab.items():
            if v > min_freq:
                result[k] = i
                i += 1
        result['oov'] = i
        json.dump(result, open(target, 'w',encoding = 'utf-8'), ensure_ascii=False)

    def gen_postag_vocab(self):
        target = os.path.join(self.data_root, 'postag_vocab.json')
        result = {'O':0,'<pad>':1}
        postag_list = ['n','f','s','t','nr','ns','nt','nw','nz','v','vd','vn','a','ad','an','d','m','q','r','p','c','u','xc','w']
        for i, (x,y) in enumerate(list(itertools.product(['B','I'],postag_list))):
            result[x + '-' + y] = i+2
        json.dump(result, open(target, 'w', encoding = 'utf-8'), ensure_ascii= False)

    def _check_valid(self, text, spo_list):
        if spo_list == []:
            return False

        if len(text) > self.hyper.max_text_len-15: # for [CLS] and [SEP]
            return False
        for t in spo_list:
            if t['object'].lower() not in text or t['subject'].lower() not in text:
                print(text)
                print(t['object'])
                print(t['subject'])
                return False
        return True

    def _read_line(self,line):
        line = line.strip("\n")
        if not line:
            return None
        instance = json.loads(line)
        text = instance['text'].lower()
        token = self.tokener._tokenize_(text)
        postag = instance['postag']
        self.word_vocab.update(token)
        if postag == []:
            return None
        postag_list = self.prepare_postag(token,postag)
        if not postag_list:
            return None

        if 'spo_list' in instance:
            spo_list = instance['spo_list']

            if not self._check_valid(text,spo_list):
                return None

            spo_list = [{
                'subject': self.tokener._tokenize_(spo['subject'].lower()),
                'predicate': spo['predicate'],
                'object': self.tokener._tokenize_(spo['object'].lower())
            } for spo in spo_list]


        result = {'text': token, 'spo_list': spo_list,
                  'bio':["O"]*len(token),'selection':[],
                  'position':[],'postag':postag_list,
                  'tag_bio':["O"]*len(token)}

        return self.prepare_bert(result)


    def prepare_postag(self, text, postag):
        postag_list = []
        word_list = []

        def _label_postag(x,pos):
            res = []
            len_ = len(x)
            for i in range(len_):
                if i == 0:
                    tag = "B-" + pos
                    res.append(tag)
                else:
                    tag = "I-" + pos
                    res.append(tag)
            return res

        for tmp in postag:
            word = self.tokener._tokenize_(tmp['word'].lower())
            pos = _label_postag(word,tmp['pos'])
            word_list.extend(word)
            postag_list.extend(pos)
        
        #assert ''.join(word_list) == ''.join(text),print(word_list) 
        if ''.join(word_list) != ''.join(text):
            print(word_list) 
            return 

        return postag_list

    def prepare_bert(self, result):

        def _index_q_list_in_k_list(q_list,k_list):
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

        def _repair(so_token,token):
            i,j  = 0,0
            while i < len(so_token) and j < len(token):
                if token[j] == so_token[i]:
                    i += 1
                    j += 1
                elif token[j].startswith(so_token[i]) or token[j].endswith(so_token[i]):
                    token[j] = so_token[i]
                    i += 1
                    j += 1
                else:
                    i = 0
                    j -= (i-1)

            if i == len(so_token):
                return token

        def _labeling_type(so_tokened,token,bio,tag_bio,is_subject = True):
            tokener_error_flag = False
            so_tokened_length = len(so_tokened)
            idx_start,idx_end = _index_q_list_in_k_list(q_list = so_tokened,k_list = token)
            if idx_start == -1 or idx_end == -1:
                if _repair(so_tokened,token):
                    token = _repair(so_tokened,token)
                    idx_start,idx_end = _index_q_list_in_k_list(q_list = so_tokened,k_list = token)
                else:
                    tokener_error_flag = True
                    print(str(so_tokened) + '@@'+str(token))
            if not tokener_error_flag:
                so_type = 'SUB' if is_subject else 'OBJ'
                bio[idx_start] = 'B'
                tag_bio[idx_start] = 'B' + '-' + so_type
                if so_tokened_length == 2:
                    bio[idx_start + 1] = 'I'
                    tag_bio[idx_start + 1] = 'I' + '-' + so_type
                elif so_tokened_length >= 3:
                    bio[idx_start + 1:idx_start + so_tokened_length] = ["I"]*(so_tokened_length-1)
                    tag_bio[idx_start + 1: idx_start + so_tokened_length] = ["I"+'-' + so_type]*(so_tokened_length - 1)
            return token,idx_start,idx_end,tokener_error_flag

        token = result['text']
        spo_list = result['spo_list']
        bio = result['bio']
        selection = result['selection']
        position = result['position']
        postag = result['postag']
        tag_bio = result['tag_bio']

        for sp in spo_list:
            p_idx = self.relation_vocab[sp['predicate']]
            token,s_idx_start,s_idx_end,flag_s = _labeling_type(sp['subject'],token,bio,tag_bio)
            token,o_idx_start,o_idx_end,flag_o = _labeling_type(sp['object'],token,bio,tag_bio,is_subject= False)

            # selection
            if not flag_s and not flag_o and not any(s == -1 for s in (s_idx_start,s_idx_end,o_idx_start,o_idx_end)):
                selection.append({
                    'subject':s_idx_end,
                    'predicate':p_idx,
                    'object':o_idx_end,
                })

                position.append({
                    'predicate': p_idx,
                    'pos':[s_idx_start,s_idx_end,o_idx_start,o_idx_end]
                })

        # for sp in spo_list:
        #     sp['subject'],sp['object'] = ''.join(sp['subject']),''.join(sp['object'])

        result = {
            'text': token,
            'spo_list': spo_list,
            'bio': bio,
            'selection': selection,
            'position':position,
            'postag':postag,
            'tag_bio':tag_bio
        }

        if not flag_s and not flag_o and len(spo_list) == len(position):
            return json.dumps(result, ensure_ascii=False)

    def _gen_one_data(self,dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root, dataset)
        with open(source, 'r',encoding='utf-8') as s, open(target, 'w',encoding='utf-8') as t:
            for line in s:
                newline = self._read_line(line)
                if newline is not None:
                    t.write(newline)
                    t.write('\n')

    def _gen_one_data_new(self,dataset):
        source = os.path.join(self.raw_data_root, dataset)
        target = os.path.join(self.data_root,'map.json')
        map_ = {}
        index = 0
        with open(source, 'r',encoding='utf-8') as s:
            for i,line in enumerate(s):
                newline = self._read_line(line)
                if newline is not None:
                    map_[i] = index
                    index += 1
        json.dump(map_, open(target, 'w', encoding = 'utf-8'), ensure_ascii= False)

    def gen_all_data(self):
        self._gen_one_data(self.hyper.train)
        self._gen_one_data(self.hyper.dev)


