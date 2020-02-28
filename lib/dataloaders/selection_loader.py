import os
import json

import torch

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

from functools import partial
from typing import Dict, List, Tuple, Set, Optional

from transformers import *

import random


class Selection_Dataset(Dataset):
    def __init__(self, hyper, dataset, is_xgb = False):
        self.hyper = hyper
        self.data_root = hyper.data_root
        self.is_xgb = is_xgb

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r',encoding='utf-8'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r',encoding='utf-8'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r',encoding='utf-8'))

        self.pso_vocab = json.load(
            open(os.path.join(self.data_root,'pso_schema.json'),'r',encoding = 'utf-8'))

        self.tag_bio_vocab = json.load(
            open(os.path.join(self.data_root,'tag_bio_vocab.json'),'r',encoding = 'utf-8')
        )
        
        if os.path.exists(os.path.join(self.data_root,'postag_vocab.json')):
            self.postag_vocab = json.load(
                open(os.path.join(self.data_root,'postag_vocab.json'),'r',encoding = 'utf-8')
            )
            self.is_pso = True
        else:
            self.is_pso = False

        self.selection_list = []
        self.text_list = []
        self.bio_list = []
        self.spo_list = []
        self.pos_list = []
        self.neg_list = []
        self.truth_list = []
        self.tag_bio_list = []
        if self.is_pso:
            self.postag_list = []

        # for bert only
        if self.hyper.cell_name == 'bert':
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                './bert-base-uncased')
        if self.hyper.cell_name == 'bert-cn':
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                './bert-base-chinese')

        for line in open(os.path.join(self.data_root, dataset), 'r',encoding='utf-8'):
            line = line.strip("\n")
            instance = json.loads(line)

            self.selection_list.append(instance['selection'])
            self.text_list.append(instance['text'])
            self.bio_list.append(instance['bio'])
            self.spo_list.append(instance['spo_list'])
            self.pos_list.append(instance['position'])
            self.tag_bio_list.append(instance['tag_bio'])

            if self.is_xgb:
                if instance.get('pos'):
                    self.truth_list.append(instance['pos'])
                self.neg_list.append(instance['neg'])
            if self.is_pso:
                self.postag_list.append(instance['postag'])

        #self._weight_count_()

    def __getitem__(self, index):
        selection = self.selection_list[index]
        text = self.text_list[index]
        bio = self.bio_list[index]
        spo = self.spo_list[index]
        pos = self.pos_list[index]

        if self.is_xgb:
            neg = self.neg_list[index]
            if len(self.truth_list):
                truth = self.truth_list[index]
            else:
                truth = []
        else:
            neg,truth = [],[]
        if self.is_pso:
            postag = self.postag_list[index]
            postag_id = self.postag2tensor(postag)
        else:
            postag_id = []

        if self.hyper.cell_name in ('bert','bert-cn'):
            schema_text_id,p_num_list,tag_bio,postag_p_id = self.schematext2tensor(text,spo,pos,postag)
            text, bio, selection = self.pad_bert(text, bio, selection)
            tokens_id = torch.tensor(
                self.bert_tokenizer.convert_tokens_to_ids(text))
        else:
            tokens_id = self.text2tensor(text)
            schema_text_id,postag_id,tag_bio,postag_p_id =[],[],[],[]

        bio_id = self.bio2tensor(bio)
        selection_id = self.selection2tensor(text, selection)
        p_id = self.p2tensor(selection)
        pos_id = self.pos2tensor(pos)
        so_id = self.so2tensor(pos)


        if self.is_xgb:
            
            #neg = [(tt['s_e'],tt['p_idx'],tt['o_e'],tt['subject'],tt['predicate'],tt['object']) for tt in neg]
            if len(neg) and neg[-1].get('distant'):
                neg = [(tt['s_s'],tt['s_e'],tt['o_s'],tt['o_e'],tt['p_idx'],tt['subject'],tt['predicate'],tt['object'],tt['distant']) for tt in neg]
            else:
                neg = [(tt['s_s'],tt['s_e'],tt['o_s'],tt['o_e'],tt['p_idx'],tt['subject'],tt['predicate'],tt['object'],0) for tt in neg]

            if len(truth):
                #truth = [(tt['s_e'],tt['p_idx'],tt['o_e'],tt['subject'],tt['predicate'],tt['object']) for tt in truth]
                truth = [(tt['s_s'],tt['s_e'],tt['o_s'],tt['o_e'],tt['p_idx'],tt['subject'],tt['predicate'],tt['object'],0) for tt in truth]

        return tokens_id, bio_id, selection_id, p_id, pos_id, len(text), spo, text, bio, index, neg, truth, so_id, schema_text_id, postag_id, tag_bio, postag_p_id


    def __len__(self):
        return len(self.text_list)

    def _weight_count_(self):
        cnt = [0]*(len(self.relation_vocab)-1)  
        for ss in self.selection_list:
            for tmp in ss:
                p = tmp['predicate']
                cnt[p] += 1

        N = float(sum(cnt))
        for i in range(len(cnt)):
            cnt[i] = N/float(cnt[i])

        weight = [0]*len(self.selection_list)
        for i in range(len(self.selection_list)):
            ss = self.selection_list[i]
            tmp = random.choice(ss)['predicate']
            weight[i] = cnt[tmp]
        
        self.weight = weight


    def pad_bert(self, text: List[str], bio: List[str], selection: List[Dict[str, int]]) -> Tuple[List[str], List[str], Dict[str, int]]:
        # for [CLS] and [SEP]
        text = ['[CLS]'] + text + ['[SEP]']
        bio = ['O'] + bio + ['O']
        selection = [{'subject': triplet['subject'] + 1, 'object': triplet['object'] +
                      1, 'predicate': triplet['predicate']} for triplet in selection]
        assert len(text) <= self.hyper.max_text_len
        text = text + ['[PAD]'] * (self.hyper.max_text_len - len(text))
        return text, bio, selection

    def text2tensor(self, text: List[str]) -> torch.tensor:
        # TODO: tokenizer
        oov = self.word_vocab['oov']
        padded_list = list(map(lambda x: self.word_vocab.get(x, oov), text))
        padded_list.extend([self.word_vocab['<pad>']] *
                           (self.hyper.max_text_len - len(text)))
        return torch.tensor(padded_list)

    def schematext2tensor(self,text,selection,position,postag):
        bio_tag_raw = ['O']*self.hyper.max_text_len
        padding_lists = []
        p_num_list = []
        p_map = {}
        postag = ['O'] + postag + ['O']
        postag_padded_list = list(map(lambda x: self.postag_vocab[x], postag))
        postag_padded_list.extend([self.postag_vocab['O']]*
                            (self.hyper.max_text_len - len(postag)))

        postag_list = []

        def _label_bio(s_s,s_e,o_s,o_e,bio):
            #bio = bio_tag_raw.copy()
            bio[s_s] = 'B' + '-' + 'SUB'
            for i in range(s_s+1,s_e+1):
                bio[i] = 'I' + '-' + 'SUB'
            bio[o_s] = 'B' + '-' + 'OBJ'
            for i in range(o_s+1,o_e+1):
                bio[i] = 'I' + '-' + 'OBJ'
            return bio

        i = 0
        for triplet in selection:
            text_raw = ['[CLS]'] + text[:] + ['[SEP]']
            len_text = len(text[:])
            p = triplet['predicate']
            if p_map.get(p):
                s_s,s_e,o_s,o_e = position[i]['pos']
                p_map[p].append((s_s,s_e,o_s,o_e))
                i += 1
                continue
            s_s,s_e,o_s,o_e = position[i]['pos']
            p_map[p] = [(s_s,s_e,o_s,o_e)]
            #s_type,o_type = self.pso_vocab[p][:2]
            schema_text = list(p)*(min(len_text,self.hyper.max_text_len - len_text-3)//len(p)) + ['[SEP]']
            #schema_text = list(s_type + ',' + p + ',' + o_type) + ['[SEP]']
            text_raw += schema_text
            assert len(text_raw) <= self.hyper.max_text_len 
            text_raw = text_raw + ['[PAD]'] * (self.hyper.max_text_len - len(text_raw))
            padded_list  = torch.tensor(
                self.bert_tokenizer.convert_tokens_to_ids(text_raw))
            padding_lists.append(padded_list)
            p_num_list.append(len(padding_lists))
            postag_list.append(torch.tensor(postag_padded_list))
            i += 1

        tag_bio_list = []
        for p,s_e in p_map.items():
            bio = bio_tag_raw.copy()
            for (s_s,s_e,o_s,o_e) in s_e:
                bio = _label_bio(s_s+1,s_e+1,o_s+1,o_e+1,bio)   
            tag_bio = list(map(lambda x: self.tag_bio_vocab[x], bio))
            tag_bio_list.append(torch.tensor(tag_bio))
            
        return torch.stack(padding_lists,0),p_num_list,torch.stack(tag_bio_list,0),torch.stack(postag_list,0)


    def bio2tensor(self, bio):
        # here we pad bio with "O". Then, in our model, we will mask this "O" padding.
        # in multi-head selection, we will use "<pad>" token embedding instead.
        padded_list = list(map(lambda x: self.bio_vocab[x], bio))
        padded_list.extend([self.bio_vocab['O']] *
                           (self.hyper.max_text_len - len(bio)))
        return torch.tensor(padded_list)

                        
    def postag2tensor(self, postag):
        postag = ['O'] + postag + ['O']
        padded_list = list(map(lambda x: self.postag_vocab[x], postag))
        padded_list.extend([self.postag_vocab['O']]*
                            (self.hyper.max_text_len - len(postag)))
        return torch.tensor(padded_list)

    def selection2tensor(self, text, selection):
        # s p o
        result = torch.zeros(
            (self.hyper.max_text_len, len(self.relation_vocab),
             self.hyper.max_text_len))
        NA = self.relation_vocab['N']
        result[:, NA, :] = 1
        for triplet in selection:

            object = triplet['object']
            subject = triplet['subject']
            predicate = triplet['predicate']

            result[subject, predicate, object] = 1
            result[subject, NA, object] = 0

        return result

    def p2tensor(self,selection):
        # p
        result = torch.zeros(len(self.relation_vocab)-1)
        for triplet in selection:
            predicate = triplet['predicate']
            result[predicate] = 1
        return result

    def pos2tensor(self,pos_list):
        result = torch.zeros(
            (self.hyper.max_text_len,len(self.relation_vocab)-1,4)
        )
    
        for triplet in pos_list:
            predicate = triplet['predicate']
            [s_s,s_e,o_s,o_e] = triplet['pos']
            result[s_s+1,predicate,0] = 1
            result[s_e+1,predicate,1] = 1
            result[o_s+1,predicate,2] = 1
            result[o_e+1,predicate,3] = 1
        return result

    def so2tensor(self,pos_list):
        p_list = []
        for triplet in pos_list:
            p = triplet['predicate']
            if p not in p_list:
                p_list.append(p)
            
        p_len = len(p_list)
        res = torch.zeros(
            (p_len,self.hyper.max_text_len,4)
        )
        for triplet in pos_list:
            p = triplet['predicate']
            b = p_list.index(p)
            [s_s,s_e,o_s,o_e] = triplet['pos']
            res[b,s_s+1,0] = 1
            res[b,s_e+1,1] = 1
            res[b,o_s+1,2] = 1
            res[b,o_e+1,3] = 1

        return res

    def postcheck_schema_transformer(self,sample,neg_score,flag):
        batch = len(neg_score)

        schema_tuple = list()
        for i in range(batch):
            raw_text = sample.text[i]
            idx = raw_text.index('[SEP]') + 1
            schema_list = list()
            for p,p_score in neg_score[i].items():
                text = raw_text.copy()
                schema_text = list(p)*(min(idx-1,self.hyper.max_text_len - idx - 1)//len(p)) + ['[SEP]']
                text[idx:idx+len(schema_text)] = schema_text
                schema_tokens = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(text))
                schema_list.append(schema_tokens)
            if schema_list != []:
                schema_tokens = torch.stack(schema_list,0)
                schema_tuple.append(schema_tokens)

        schema_tokens = torch.cat(tuple(schema_tuple),dim = 0)
        sample.schema_token_p = schema_tokens
        sample.pso_score = neg_score
        sample.pso_flag = flag

    def schema_transformer(self,output,sample):
        p_decode = output['p_decode']
        p_triplet = output['p_schema_triples']
        batch = len(p_decode)
        output_p = [[] for _ in range(batch)]
        postag_list = []
        p_score = [[] for _ in range(batch)]
        r_p_idx_list = []


        """
        schema_tuple = list()
        for i in range(batch):
            text = sample.text[i]
            idx = text.index('[SEP]')+1
            p_len = len(p_triplet[i])
            schema_list = list()
            for tmp in p_triplet[i]:
                tokens = tokens_raw[i,:]
                s_type,p,o_type = tmp['s_type'],tmp['predicate'],tmp['o_type']
                output_p[i].append(p)
                schema_text = list(p)*(min(idx-1,self.hyper.max_text_len - idx -1)//len(p))
                #schema_text = list(s_type + ',' + p + ',' + o_type) + ['[SEP]']
                text[idx:idx+len(schema_text)] = schema_text
                print(text)
                schema_tokens = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(schema_text))
                tokens[idx:idx+len(schema_text)] = schema_tokens
                schema_list.append(tokens)
            if schema_list != []:
                schema_tokens = torch.stack(schema_list,0)
                schema_tuple.append(schema_tokens)
        
        schema_tokens = torch.cat(tuple(schema_tuple),dim = 0)
        sample.schema_token_p = schema_tokens
        sample.output_p = output_p
        """
        schema_tuple = list()
        for i in range(batch):
            raw_text = sample.text[i]
            postag_id = sample.postag_id[i,:].unsqueeze(0)
            idx = raw_text.index('[SEP]')+1
            p_len = len(p_triplet[i])
            postag_id = postag_id.expand(p_len,-1)
            schema_list = list()
            for tmp in p_triplet[i]:
                text = raw_text.copy()
                s_type,p,o_type,score = tmp['s_type'],tmp['predicate'],tmp['o_type'],tmp['p_score']
                output_p[i].append(p)
                r_p_idx_list.append(self.relation_vocab[p])
                p_score[i].append(score)
                schema_text = list(p)*(min(idx-1,self.hyper.max_text_len - idx -1)//len(p)) + ['[SEP]']
                text[idx:idx+len(schema_text)] = schema_text
                schema_tokens = torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(text))
                schema_list.append(schema_tokens)
            if schema_list != []:
                schema_tokens = torch.stack(schema_list,0)
                schema_tuple.append(schema_tokens)
                postag_list.append(postag_id)
        
        schema_tokens = torch.cat(tuple(schema_tuple),dim = 0)
        postag_p_id = torch.cat(tuple(postag_list),dim = 0)
        sample.schema_token_p = schema_tokens
        sample.output_p = output_p
        sample.postag_p_id = postag_p_id
        sample.p_score = p_score
        sample.reverse_p_id = torch.tensor(r_p_idx_list)
 
class Batch_reader(object):
    def __init__(self, data):
        transposed_data = list(zip(*data))
        # tokens_id, bio_id, selection_id, p_id, pos_id, len(text), spo, text, bio, index, neg, truth, so, schema_text_id, postag,tag_bio,postag_p_id

        self.tokens_id = pad_sequence(transposed_data[0], batch_first=True)
        self.bio_id = pad_sequence(transposed_data[1], batch_first=True)
        self.selection_id = torch.stack(transposed_data[2], 0)
        self.p_id = pad_sequence(transposed_data[3], batch_first= True)
        self.pos_id = torch.stack(transposed_data[4],0)
        self.length = transposed_data[5]

        self.spo_gold = transposed_data[6]
        self.text = transposed_data[7]
        self.bio = transposed_data[8]
        self.index = transposed_data[9]
        self.neg_list = transposed_data[10]
        self.pos_list = transposed_data[11]
        
        def cat(tensors):
            return torch.cat(tuple(tensors),dim = 0)

        self.so_id = cat(transposed_data[12])
        self.flag = False
        if type(transposed_data[13][0]) != list:
            self.flag = True
            self.schema_text_id = cat(transposed_data[13])
            self.postag_id = torch.stack(transposed_data[14],0)
            self.tag_bio = cat(transposed_data[15])
            self.postag_p_id = cat(transposed_data[16])

            # reverse p_id
            self.reverse_p_id = torch.where(self.p_id == 1)[1]

    def pin_memory(self):
        self.tokens_id = self.tokens_id.pin_memory()
        self.bio_id = self.bio_id.pin_memory()
        self.selection_id = self.selection_id.pin_memory()
        self.p_id = self.p_id.pin_memory()
        self.pos_id = self.pos_id.pin_memory()
        self.so_id = self.so_id.pin_memory()
        if self.flag:
            self.schema_text_id = self.schema_text_id.pin_memory()
            self.postag_id = self.postag_id.pin_memory()
            self.tag_bio = self.tag_bio.pin_memory()
            self.postag_p_id = self.postag_p_id.pin_memory()
            self.reverse_p_id = self.reverse_p_id.pin_memory()
        return self


def collate_fn(batch):
    return Batch_reader(batch)


Selection_loader = partial(DataLoader, collate_fn=collate_fn, pin_memory=True)
