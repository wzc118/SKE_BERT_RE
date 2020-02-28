import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import copy

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from transformers import *
import numpy as np

class BaseModel(nn.Module):
    def __init__(self,hyper):
        super(BaseModel, self).__init__()

        self.hyper = hyper
        self.data_root = hyper.data_root

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r',encoding='utf-8'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r',encoding='utf-8'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r',encoding='utf-8'))

        self.pso_vocab = json.load(
            open(os.path.join(self.data_root,'pso_schema.json'),'r',encoding = 'utf-8'))  

        self.id2bio = {v: k for k, v in self.bio_vocab.items()}

        self.reversed_relation_vocab = {
            v: k
            for k, v in self.relation_vocab.items()
        }

        self.encoder = BertModel.from_pretrained('./bert-base-chinese',output_attentions=False)

        for name, param in self.encoder.named_parameters():
            if '11' in name or '10' in name or '9' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            './bert-base-chinese') 

        self.drop = nn.Dropout(0.1)

    def forward(self,sample,is_train):
        pass

class P_Model_new(BaseModel):
    def __init__(self,hyper,is_xgb = False):
        super(P_Model_new,self).__init__(hyper)
        self.p_linear = nn.Linear(768,len(self.relation_vocab)-1)
        self.is_xgb = is_xgb
        
    def p_decode(self,p_logits):
        """
        p_logits: [B,H]
        """
        batch_num = p_logits.size(0)
        result = [[] for _ in range(batch_num)]
        p_words = [[] for _ in range(batch_num)]

        logits = torch.sigmoid(p_logits)
        p_tags = logits > self.hyper.threshold
        
        idx = torch.nonzero(p_tags.cpu())
        for i in range(idx.size(0)):
            b,p = idx[i].tolist()
            predicate = self.reversed_relation_vocab[p]
            # find schema and add text
            s_type,o_type = self.pso_vocab[predicate][:2]
            p_score = logits[b,p].cpu().item()

            triplet = {
                's_type':s_type,
                'predicate':predicate,
                'o_type':o_type,
                'p_score':p_score
            }
            result[b].append(triplet)
            p_words[b].append(predicate)

        score = logits.cpu().numpy().tolist()

        return result,p_words,score

    def FocalLoss(self,p_logits,p_gold,alpha = 0.25, gamma = 2):
        bce_loss = F.binary_cross_entropy_with_logits(p_logits,p_gold,reduction='none')
        pt = torch.exp(-bce_loss)
        F_loss = alpha*(1-pt)**gamma*bce_loss*100
        loss = F_loss.mean()
        return loss

    @staticmethod
    def description(epoch,epoch_num,output):
        return "L: {:.2f}, epoch: {}/{}:".format(
            output['loss'].item(),epoch,epoch_num)

    def _init_spo_search(self,spo_maker):
        self.spo_maker = spo_maker

    def pos_neg_score(self,p_logits,pos_neg):

        p_logits = torch.sigmoid(p_logits)
        res = [dict() for _ in range(len(pos_neg))]

        for i in range(len(pos_neg)):
            neg = pos_neg[i]
            p_set = set()
            res[i] = dict()
            for _,_,_,_,p_idx,_,p,_,_ in neg:
                if p not in p_set:
                    p_score = p_logits[i,int(p_idx)].cpu().item()
                    res[i][p] = p_score
                    p_set.add(p)

        return res

    def forward(self, sample, is_train):
        output = {}

        tokens = sample.tokens_id.to(self.hyper.device)
        p_gold_idx = sample.p_id.to(self.hyper.device)

        text_list = sample.text

        if self.is_xgb:
            neg_list = sample.neg_list
            pos_list = sample.pos_list

        notpad = tokens != self.bert_tokenizer.encode('[PAD]')[0]
        notcls = tokens != self.bert_tokenizer.encode('[CLS]')[0]
        notsep = tokens != self.bert_tokenizer.encode('[SEP]')[0]
        mask = notpad 

        pool_o = self.encoder(tokens, attention_mask=mask)[1]

        if is_train:
            pool_o = self.drop(pool_o)
            pool_o = self.p_linear(pool_o)
            p_loss = self.FocalLoss(pool_o,p_gold_idx)
            output['loss'] = p_loss
        else:
            pool_o = self.p_linear(pool_o)
            if not self.is_xgb:
                output['p_schema_triples'],output['p_decode'],output['p_logits'] = self.p_decode(pool_o)

                p_gold = [[] for _ in range(len(text_list))]
                p_idx = torch.nonzero(sample.p_id)

                for i in range(p_idx.size(0)):
                    b,p = p_idx[i].tolist()
                    p_gold[b].append(self.reversed_relation_vocab[p])
                    output['p_golden'] = p_gold

        if self.is_xgb:
            p_neg_score = self.pos_neg_score(pool_o,neg_list)
            output['neg_score'] = p_neg_score
            if len(pos_list[0]) > 0:
                p_pos_score = self.pos_neg_score(pool_o,pos_list)
                output['pos_score'] = p_pos_score

        output['description'] = partial(self.description, output=output)

        return output


class SO_Model_New(BaseModel):
    def __init__(self,hyper,is_xgb):
        super(SO_Model_New, self).__init__(hyper)
        
        self.is_xgb = is_xgb

        self.tag_bio_vocab = json.load(
            open(os.path.join(self.data_root, 'tag_bio_vocab.json'), 'r', encoding = 'utf-8'))


        self.so_linear = nn.Linear(768,len(self.tag_bio_vocab))
        self.p_linear = nn.Linear(768,len(self.relation_vocab)-1)


    def get_postag(self,sample):

        postag_list = []

        postag_id = sample.postag_id.cpu().numpy()

        postag_vocab = json.load(
            open(os.path.join(self.data_root, 'postag_vocab.json'),'r',encoding = 'utf-8')
        )

        reversed_postag_vocab = {
            v:k
            for k,v in postag_vocab.items()
        }

        b = postag_id.shape[0]

        for i in range(b):
            postag_res = list(postag_id[i,:])
            postag_list.append([reversed_postag_vocab[k] for k in postag_res])
        
        return postag_list

    @staticmethod
    def description(epoch,epoch_num,output):
        return "L: {:.2f}, L_p:{:.2f}, L_so:{:.2f}, epoch: {}/{}:".format(
            output['loss'].item(),output['p_loss'].item(),
            output['so_loss'].item(),epoch,epoch_num
            )

    def _init_spo_search(self,spo_maker):
        self.spo_maker = spo_maker

    def spo_search(self,text_list,result):

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

        for i in range(len(text_list)):
            text = ''.join(text_list[i])
            res = self.spo_maker.extract_items_list(text)
            if len(res) > 0:
                res_new = []
                for tmp in res:
                    s,o = tmp['subject'],tmp['object']
                    s_s,s_e = _find_end_idx(s,text_list[i])
                    o_s,o_e = _find_end_idx(o,text_list[i])
                    if s_e != -1 and o_e != -1:
                        tmp['s_s'],tmp['s_e'],tmp['o_s'],tmp['o_e'],tmp['p_idx'] = s_s,s_e,o_s,o_e,self.relation_vocab[tmp['predicate']]
                        tmp['subject'],tmp['object'] = ''.join(s),''.join(o)
                        res_new.append(tmp.copy())
                result[i].extend(res_new)

    def so_loss(self,so_logits,so_gold,mask):
        weight = [1,1,2,2,2,2,1,1]
        class_weight = torch.FloatTensor(weight).to(self.hyper.device)
        loss_fct = nn.CrossEntropyLoss(weight= class_weight)
        active_loss = mask.view(-1) == 1
        active_logits = so_logits.view(-1,len(self.tag_bio_vocab))[active_loss]
        active_labels = so_gold.view(-1)[active_loss]
        loss = loss_fct(active_logits,active_labels)

        return loss

    def p_loss(self,p_logits,p_gold):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(p_logits.view(-1,len(self.relation_vocab)-1),p_gold.view(-1))
        return loss

    def pos_neg_score(self,so_logits,p_score,pos_neg,postag_gold):
        
        preds_ = F.softmax(so_logits,dim = -1).cpu().numpy() #[B,S]
        preds = so_logits.cpu().numpy() #[B,S,tag_len]
        preds = np.argmax(preds,axis = 2)

        map_,res = {},0
        for i in range(len(pos_neg)):
            tmp = pos_neg[i]
            for j in range(len(tmp)):
                map_[res+j] = i
            res += len(tmp)

        def _score_(p_s,s_s,o_s):
            return (p_s + sum(s_s)/len(s_s) + sum(o_s)/len(o_s))/3

        def _score_mul(p_s,s_s,o_s):
            return p_s*(sum(s_s)/len(s_s))*(sum(o_s)/len(o_s))

        def _check_(s,e,postag):
            prev = None
            if postag[e+1].startswith('B') or postag[e+1] == 'O':
                pass
            else:
                return False

            for i in range(s,e+1):
                if postag[i].startswith("B"):
                    prev = postag[i].split('-')[-1]
                elif postag[i].startswith("I"):
                    if not prev or postag[i].split('-')[-1] != prev:
                        return False
            return True


        res = [[] for _ in range(len(pos_neg))]
        for i in range(len(pos_neg)):
            neg = pos_neg[i]
            postag = postag_gold[i]
            for s_s,s_e,o_s,o_e,p_idx,s,p,o,distant in neg:
                s_score,o_score = [],[]
                for j in range(s_s,s_e+1):
                    s_score.append(preds_[map_[i]][j][preds[map_[i]][j]])
                for j in range(o_s,o_e+1):
                    o_score.append(preds_[map_[i]][j][preds[map_[i]][j]])
                p_s = p_score[i][p]
                score = _score_(p_s,s_score,o_score)

                if _check_(s_s,s_e,postag) and _check_(o_s,o_e,postag):
                    seg = 1
                else:
                    seg = 0

                triplet = {
                    'object':o,
                    'predicate':p,
                    'subject':s,
                    'score':_score_mul(p_s,s_score,o_score),
                    'distant':distant,
                    'seg':seg,
                    'p_score':p_s,
                    's_score':sum(s_score)/len(s_score),
                    'o_score':sum(o_score)/len(o_score)
                }

                res[i].append(triplet)
        
        return res

    def so_decode(self,so_logits,text,output_p,p_score):


        def _check(s,e,postag):
            prev = None
            if postag[e+1].startswith('B') or postag[e+1] == 'O':
                pass
            else:
                return False

            for i in range(s,e+1):
                if postag[i].startswith("B"):
                    prev = postag[i].split('-')[-1]
                elif postag[i].startswith("I"):
                    if not prev or postag[i].split('-')[-1] != prev:
                        return False
            return True

        result = [[] for _ in range(len(output_p))]
        
        output_p_flatten = sum(output_p,[])
        p_score_flatten = sum(p_score,[])
        p_num_list,text_map,res = [],{},0
        for i in range(len(output_p)):
            tmp = output_p[i]
            p_num_list.append(len(tmp))
            for j in range(len(tmp)):
                text_map[res+j] = i
            res += len(tmp)

        b,s = so_logits.size(0),so_logits.size(1)
        preds_ = F.softmax(so_logits,dim = -1).cpu().numpy()
        preds = so_logits.cpu().numpy() #[B,S,tag_len]
        preds = np.argmax(preds,axis = 2)

        reversed_tag_bio_vocab = {v: k for k, v in self.tag_bio_vocab.items()}

        for i in range(b):
            entity_list,entity_part_list,score_list,score_part_list,position_list,position_part_list = [],[],[],[],[],[]
            text_i = text[text_map[i]]
            max_ids = text_i.index('[SEP]')
            p_score = p_score_flatten[i]
            postag = self.postag_gold[text_map[i]]
            for j in range(1,max_ids):
                tags = reversed_tag_bio_vocab[preds[i][j]]
                if tags == "O":
                    if len(entity_part_list) > 0:
                        entity_list.append(entity_part_list)
                        entity_part_list = []
                        score_list.append(score_part_list)
                        score_part_list = []
                        position_list.append(position_part_list)
                        position_part_list = []
                if tags.startswith("B-"):
                    if len(entity_part_list) > 0:
                        entity_list.append(entity_part_list)
                        entity_part_list = []
                        score_list.append(score_part_list)
                        score_part_list = []
                        position_list.append(position_part_list)
                        position_part_list = []
                    entity_part_list.append(tags)
                    entity_part_list.append(text_i[j])
                    score_part_list.append(preds_[i][j][preds[i][j]])
                    position_part_list.append(j)
                    if j == max_ids-1:
                        entity_list.append(entity_part_list)
                        score_list.append(score_part_list)
                        position_list.append(position_part_list)
                if tags.startswith("I-"):
                    if len(entity_part_list) > 0:
                        entity_part_list.append(text_i[j])
                        score_part_list.append(preds_[i][j][preds[i][j]])
                        position_part_list.append(j)
                        if j == max_ids-1:
                            entity_list.append(entity_part_list)
                            score_list.append(score_part_list)
                            position_list.append(position_part_list)

            def _score_(p_s,s_s,o_s):
                return (p_s + sum(s_s)/len(s_s) + sum(o_s)/len(o_s))/3

            def _score_mul(p_s,s_s,o_s):
                return p_s*(sum(s_s)/len(s_s))*(sum(o_s)/len(o_s))

            # merge
            subject,object,s_s,s_e,o_s,o_e = None,None,0,0,0,0
            for e,s,pos in zip(entity_list,score_list,position_list):
                e_content= ""
                e_type = None
                for idx, e_part in enumerate(e):
                    if idx == 0:
                        e_type = e_part
                        if e_type[:2] not in ["B-","I-"]:
                            break
                    else:
                        e_content += e_part

                if e_type == 'B-SUB':
                    subject = e_content
                    s_score = s
                    s_s = pos[0]
                    s_e = pos[-1]

                elif e_type == 'B-OBJ':
                    object = e_content
                    o_score = s
                    o_s = pos[0]
                    o_e = pos[-1]

                prediacte = output_p_flatten[i]

                if _check(s_s,s_e,postag) and _check(o_s,o_e,postag):
                    seg = 1
                else:
                    seg = 0

                if subject and object:
                    triplet = {
                        'object':object,
                        'predicate':prediacte,
                        'subject':subject,
                        'distant': 0,
                        'seg':seg,
                        'score': _score_mul(p_score,s_score,o_score),
                        's_s':s_s,
                        's_e':s_e,
                        'o_s':o_s,
                        'o_e':o_e,
                        'p_idx':self.relation_vocab[prediacte]
                    }
                    result[text_map[i]].append(triplet)
        
        self.spo_search(text,result)

        for i in range(len(result)):
            result[i] = self.filter(result[i])

        return result

    def filter(self,triplet):
        p1 = p2 = triplet

        def _check(tmp,gold):
            s,p,o = tmp['subject'],tmp['predicate'],tmp['object']
            for g in gold:
                if p == g['predicate'] and len(g['subject']) > len(s) and g['subject'].endswith(s) and g['object'] == o:
                    return True
                if p == g['predicate'] and len(g['object']) > len(o) and g['subject'].endswith(s) and g['subject'] == s:
                    return True
            return False

        idx = 0
        for tmp in p1:
            if _check(tmp,p2):
                p1.pop(idx)
            else:
                idx += 1
        return p1

    def forward(self, sample, is_train):
        output = {}

        if is_train:
            schema_tokens = sample.schema_text_id.to(self.hyper.device)
        else:
            schema_tokens = sample.schema_token_p.to(self.hyper.device)

        tag_bio = sample.tag_bio.to(self.hyper.device)
        p_gold_idx = sample.reverse_p_id.to(self.hyper.device)

        if self.is_xgb:
            neg_list = sample.neg_list
            pos_list = sample.pos_list


        text_list = sample.text
        spo_gold = sample.spo_gold
        if not is_train and not self.is_xgb:
            p_score = sample.p_score

        notpad = schema_tokens != self.bert_tokenizer.encode('[PAD]')[0]
        notcls = schema_tokens != self.bert_tokenizer.encode('[CLS]')[0]
        notsep = schema_tokens != self.bert_tokenizer.encode('[SEP]')[0]
        bert_mask = notpad 
        mask = notpad & notcls & notsep

        o,pool_o = self.encoder(schema_tokens, attention_mask = bert_mask)[:2]
        #self.attn = self.encoder(schema_tokens, attention_mask = bert_mask)[-1][-3:]

        if is_train:
            o = self.drop(o)
            pool_o = self.drop(pool_o)
        pool_o = self.p_linear(pool_o)
        o = self.so_linear(o)

        if is_train:
            so_loss = self.so_loss(o,tag_bio,mask)
            p_loss = self.p_loss(pool_o,p_gold_idx)

            loss = 0.5*p_loss + so_loss
            output['p_loss'] = p_loss
            output['so_loss'] = so_loss
            output['loss'] = loss

        elif not is_train and not self.is_xgb:
            # decode
            self.postag_gold = self.get_postag(sample)
            output['selection_triplets'] = self.so_decode(o,text_list,sample.output_p,p_score)
            #output['selection_triplets'] = self.so_decode_test(o,pool_o,text_list,sample.output_p)

            for tmp in spo_gold:
                for sp in tmp:
                    sp['subject'],sp['object'] = ''.join(sp['subject']),''.join(sp['object'])

            output['spo_gold'] = spo_gold


        if self.is_xgb:
            self.postag_gold = self.get_postag(sample)
            if not sample.pso_flag:
                p_neg_score = self.pos_neg_score(o,sample.pso_score,neg_list,self.postag_gold)
                output['neg_list'] = p_neg_score
            else:
                p_pos_score = self.pos_neg_score(o,sample.pso_score,pos_list,self.postag_gold)
                output['pos_list'] = p_pos_score

        output['description'] = partial(self.description, output=output)

        return output








        

        
        
        




