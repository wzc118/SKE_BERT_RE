import torch
import torch.nn as nn
import torch.nn.functional as F

import json
import os
import copy
import random

from typing import Dict, List, Tuple, Set, Optional
from functools import partial

from torchcrf import CRF
from transformers import *


class MultiHeadSelection(nn.Module):
    def __init__(self, hyper,is_xgb = False) -> None:
        super(MultiHeadSelection, self).__init__()

        self.hyper = hyper
        self.data_root = hyper.data_root
        self.gpu = hyper.gpu
        self.is_xgb = is_xgb

        self.word_vocab = json.load(
            open(os.path.join(self.data_root, 'word_vocab.json'), 'r',encoding='utf-8'))
        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r',encoding='utf-8'))
        self.bio_vocab = json.load(
            open(os.path.join(self.data_root, 'bio_vocab.json'), 'r',encoding='utf-8'))
        self.id2bio = {v: k for k, v in self.bio_vocab.items()}

        self.word_embeddings = nn.Embedding(num_embeddings=len(
            self.word_vocab),
            embedding_dim=hyper.emb_size)

        self.relation_emb = nn.Embedding(num_embeddings=len(
            self.relation_vocab),
            embedding_dim=hyper.rel_emb_size)
        # bio + pad
        self.bio_emb = nn.Embedding(num_embeddings=len(self.bio_vocab),
                                    embedding_dim=hyper.bio_emb_size)

        if hyper.cell_name == 'gru':
            self.encoder = nn.GRU(hyper.emb_size,
                                  hyper.hidden_size,
                                  bidirectional=True,
                                  batch_first=True)
        elif hyper.cell_name == 'lstm':
            self.encoder = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)
        elif hyper.cell_name == 'bert' or hyper.cell_name == 'bert-cn':
            self.post_lstm = nn.LSTM(hyper.emb_size,
                                   hyper.hidden_size,
                                   bidirectional=True,
                                   batch_first=True)
            if hyper.cell_name == 'bert':
                self.encoder = BertModel.from_pretrained('./bert-base-uncased')
            else:
                self.encoder = BertModel.from_pretrained('./bert-base-chinese',output_attentions=False)
            for name, param in self.encoder.named_parameters():
                if '11' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                    # print(name, param.size())
        else:
            raise ValueError('cell name should be gru/lstm/bert!')

        if hyper.activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif hyper.activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('unexpected activation!')

        self.tagger = CRF(len(self.bio_vocab) - 1, batch_first=True)

        self.selection_u = nn.Linear(hyper.hidden_size + hyper.bio_emb_size,
                                     hyper.rel_emb_size)
        self.selection_v = nn.Linear(hyper.hidden_size + hyper.bio_emb_size,
                                     hyper.rel_emb_size)
        self.selection_uv = nn.Linear(2 * hyper.rel_emb_size,
                                      hyper.rel_emb_size)
        self.emission = nn.Linear(hyper.hidden_size, len(self.bio_vocab) - 1)

        self.bert2hidden = nn.Linear(768, hyper.hidden_size)
        # for bert_lstm
        # self.bert2hidden = nn.Linear(768, hyper.emb_size)

        if self.hyper.cell_name == 'bert':

            self.bert_tokenizer = BertTokenizer.from_pretrained(
                './bert-base-uncased')
        
        if self.hyper.cell_name == 'bert-cn':
            self.bert_tokenizer = BertTokenizer.from_pretrained(
                './bert-base-chinese')            

        # self.accuracy = F1Selection()

    def inference(self, mask, text_list, decoded_tag, selection_logits):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab),
                              -1)  # batch x seq x rel x seq
        selection_tags = (torch.sigmoid(selection_logits) *
                          selection_mask.float()) > self.hyper.threshold

        #selection_triplets = self.selection_decode(text_list, decoded_tag,
                                                   #selection_tags)

        selection_triplets = self.selection_decode_new(text_list, decoded_tag,
                                                   selection_tags,selection_logits,self.postag_gold)

        return selection_triplets

    def masked_BCEloss(self, mask, selection_logits, selection_gold):
        selection_mask = (mask.unsqueeze(2) *
                          mask.unsqueeze(1)).unsqueeze(2).expand(
                              -1, -1, len(self.relation_vocab),
                              -1)  # batch x seq x rel x seq
        selection_loss = F.binary_cross_entropy_with_logits(selection_logits,
                                                            selection_gold,
                                                            reduction='none')
        selection_loss = selection_loss.masked_select(selection_mask).sum()
        selection_loss /= mask.sum()
        return selection_loss

    @staticmethod
    def description(epoch, epoch_num, output):
        return "L: {:.2f}, L_crf: {:.2f}, L_selection: {:.2f}, epoch: {}/{}:".format(
            output['loss'].item(), output['crf_loss'].item(),
            output['selection_loss'].item(), epoch, epoch_num)

    def _init_spo_search(self,spo_maker):
        self.spo_maker = spo_maker

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

    def forward(self, sample, is_train: bool) -> Dict[str, torch.Tensor]:
        
        """
        tokens = sample.tokens_id.cuda(self.gpu)
        selection_gold = sample.selection_id.cuda(self.gpu)
        bio_gold = sample.bio_id.cuda(self.gpu)
        """
        tokens = sample.tokens_id.to(self.hyper.device)
        selection_gold = sample.selection_id.to(self.hyper.device)
        bio_gold = sample.bio_id.to(self.hyper.device)

        text_list = sample.text
        spo_gold = sample.spo_gold

        bio_text = sample.bio

        if self.is_xgb:
            neg_list = sample.neg_list
            pos_list =  sample.pos_list

        if self.hyper.cell_name in ('gru', 'lstm'):
            mask = tokens != self.word_vocab['<pad>']  # batch x seq
            bio_mask = mask
        elif self.hyper.cell_name in ('bert','bert-cn'):
            notpad = tokens != self.bert_tokenizer.encode('[PAD]')[0]
            notcls = tokens != self.bert_tokenizer.encode('[CLS]')[0]
            notsep = tokens != self.bert_tokenizer.encode('[SEP]')[0]
            mask = notpad & notcls & notsep
            bio_mask = notpad & notsep  # fst token for crf cannot be masked
        else:
            raise ValueError('unexpected encoder name!')

        if self.hyper.cell_name in ('lstm', 'gru'):
            embedded = self.word_embeddings(tokens)
            o, h = self.encoder(embedded)

            o = (lambda a: sum(a) / 2)(torch.split(o,
                                                   self.hyper.hidden_size,
                                                   dim=2))
        elif self.hyper.cell_name in ('bert','bert-cn'):
            # with torch.no_grad():
            o = self.encoder(tokens, attention_mask=mask)[
                0]  # last hidden of BERT
            # o = self.activation(o)
            # torch.Size([16, 310, 768])
            #self.attn = self.encoder(tokens, attention_mask = mask)[-1][-1]
            o = self.bert2hidden(o)

            # below for bert+lstm
            # o, h = self.post_lstm(o)

            # o = (lambda a: sum(a) / 2)(torch.split(o,
            #                                        self.hyper.hidden_size,
            #                                        dim=2))
        else:
            raise ValueError('unexpected encoder name!')
        emi = self.emission(o)

        output = {}

        crf_loss = 0

        if is_train:
            crf_loss = -self.tagger(emi, bio_gold,
                                    mask=bio_mask, reduction='mean')
        elif not self.is_xgb:
            decoded_tag = self.tagger.decode(emissions=emi, mask=bio_mask)

            output['decoded_tag'] = [list(map(lambda x : self.id2bio[x], tags)) for tags in decoded_tag]
            output['gold_tags'] = bio_text

            temp_tag = copy.deepcopy(decoded_tag)
            for line in temp_tag:
                line.extend([self.bio_vocab['<pad>']] *
                            (self.hyper.max_text_len - len(line)))
            #bio_gold = torch.tensor(temp_tag).cuda(self.gpu)
            bio_gold = torch.tensor(temp_tag).to(self.hyper.device)

        tag_emb = self.bio_emb(bio_gold)

        o = torch.cat((o, tag_emb), dim=2)

        # forward multi head selection
        B, L, H = o.size()
        u = self.activation(self.selection_u(o)).unsqueeze(1).expand(B, L, L, -1)
        v = self.activation(self.selection_v(o)).unsqueeze(2).expand(B, L, L, -1)
        uv = self.activation(self.selection_uv(torch.cat((u, v), dim=-1)))

        # correct one
        selection_logits = torch.einsum('bijh,rh->birj', uv,
                                        self.relation_emb.weight)

        # use loop instead of matrix
        # selection_logits_list = []
        # for i in range(self.hyper.max_text_len):
        #     uvi = uv[:, i, :, :]
        #     sigmoid_input = uvi
        #     selection_logits_i = torch.einsum('bjh,rh->brj', sigmoid_input,
        #                                         self.relation_emb.weight).unsqueeze(1)
        #     selection_logits_list.append(selection_logits_i)
        # selection_logits = torch.cat(selection_logits_list,dim=1)


        if not is_train and not self.is_xgb:
            self.postag_gold = self.get_postag(sample)
            output['selection_triplets'] = self.inference(
                mask, text_list, decoded_tag, selection_logits)

            for tmp in spo_gold:
                for sp in tmp:
                    sp['subject'],sp['object'] = ''.join(sp['subject']),''.join(sp['object'])

            output['spo_gold'] = spo_gold

        selection_loss = 0
        if is_train:
            selection_loss = self.masked_BCEloss(mask, selection_logits,
                                                 selection_gold)

        # postcheck
        if self.is_xgb:
            self.postag_gold = self.get_postag(sample)
            neg_res = self.pos_neg_score(selection_logits,neg_list,self.postag_gold)
            output['neg_list'] = neg_res
            if len(pos_list[0]) > 0:
                pos_res = self.pos_neg_score(selection_logits,pos_list,self.postag_gold)
                output['pos_list'] = pos_res

        loss = crf_loss + selection_loss
        output['crf_loss'] = crf_loss
        output['selection_loss'] = selection_loss
        output['loss'] = loss

        output['description'] = partial(self.description, output=output)
        return output

    def selection_decode(self, text_list, sequence_tags,
                         selection_tags: torch.Tensor
                         ) -> List[List[Dict[str, str]]]:
        reversed_relation_vocab = {
            v: k
            for k, v in self.relation_vocab.items()
        }

        reversed_bio_vocab = {v: k for k, v in self.bio_vocab.items()}

        text_list = list(map(list, text_list))

        def find_entity(pos, text, sequence_tags):
            entity = []

            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                entity = list(reversed(temp_entity))
            return ''.join(entity)

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        idx = torch.nonzero(selection_tags.cpu())
        for i in range(idx.size(0)):
            b, s, p, o = idx[i].tolist()

            predicate = reversed_relation_vocab[p]
            if predicate == 'N':
                continue
            tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
            try:
                object = find_entity(o, text_list[b], tags)
            except:
                print(o)
                print(text_list[b])
                print(tags)
            subject = find_entity(s, text_list[b], tags)

            assert object != '' and subject != ''

            triplet = {
                'object': object,
                'predicate': predicate,
                'subject': subject
            }
            result[b].append(triplet)

        for i in range(len(result)):
            result[i] = self.filter(result[i])

        return result


    def selection_decode_new(self,text_list,sequence_tags,
                            selection_tags,selection_logits,postag_gold):
        reversed_relation_vocab = {
            v: k
            for k, v in self.relation_vocab.items()
        }

        reversed_bio_vocab = {v: k for k, v in self.bio_vocab.items()}

        text_list = list(map(list, text_list))

        def find_entity(pos, text, sequence_tags):
            entity = []

            if sequence_tags[pos] in ('B', 'O'):
                entity.append(text[pos])
            else:
                temp_entity = []
                while sequence_tags[pos] == 'I':
                    temp_entity.append(text[pos])
                    pos -= 1
                    if pos < 0:
                        break
                    if sequence_tags[pos] == 'B':
                        temp_entity.append(text[pos])
                        break
                entity = list(reversed(temp_entity))
            return ''.join(entity)

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

        batch_num = len(sequence_tags)
        result = [[] for _ in range(batch_num)]
        idx = torch.nonzero(selection_tags.cpu())
        selection_logits = torch.sigmoid(selection_logits)

        for i in range(idx.size(0)):
            b,s,p,o = idx[i].tolist()
            predicate = reversed_relation_vocab[p]
            postag = postag_gold[b]
            if predicate == 'N':
                continue
            score = selection_logits[b,s,p,o].cpu().item()
            tags = list(map(lambda x: reversed_bio_vocab[x], sequence_tags[b]))
            object = find_entity(o, text_list[b], tags)
            subject = find_entity(s, text_list[b], tags)
            s_s,s_e,o_s,o_e = s-len(subject)+1,s,o-len(object)+1,o
            if _check_(s_s,s_e,postag) and _check_(o_s,o_e,postag):
                seg = 1
            else:
                seg = 0

            assert object != '' and subject != ''

            triplet = {
                'object':object,
                'predicate': predicate,
                'subject':subject,
                'score':score,
                'distant': 0,
                'seg':seg
            } 

            if self.is_xgb:
                triplet['label'] = 1

            result[b].append(triplet)
        

        # distant spo search
        if not self.is_xgb:
            self.spo_search(text_list,result,selection_logits,postag_gold)

        for i in range(len(result)):
            result[i] = self.filter(result[i])
            
        return result 

    def pos_neg_score(self, selection_logits,pos_neg,postag_gold):
        
        selection_logits = torch.sigmoid(selection_logits)

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
                score = selection_logits[i,s_e,p_idx,o_e].cpu().item()

                if _check_(s_s,s_e,postag) and _check_(o_s,o_e,postag):
                    seg = 1
                else:
                    seg = 0

                triplet = {
                    'object':o,
                    'predicate':p,
                    'subject':s,
                    'score':score,
                    'distant':distant,
                    'seg':seg
                } 
                res[i].append(triplet)
        return res 

        
    def spo_search(self,text_list,result,selection_logits,postag_gold):

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
            postag = postag_gold[i]
            res = self.spo_maker.extract_items_list(text)
            res_new = []
            if len(res) > 0:
                for triplet in res:
                    s,o,p = triplet['subject'],triplet['object'],self.relation_vocab[triplet['predicate']]
                    s_s,s_e = _find_end_idx(s,text_list[i])
                    o_s,o_e = _find_end_idx(o,text_list[i])
                    if s_e != -1 and o_e != -1:
                        score = selection_logits[i,s_e,p,o_e].cpu().item()
                        triplet['score'] = score
                        if _check_(s_s,s_e,postag) and _check_(o_s,o_e,postag):
                            seg = 1
                        else:
                            seg = 0
                        triplet['seg'] = seg
                        triplet['subject'] = ''.join(s)
                        triplet['object'] = ''.join(o)
                        res_new.append(triplet.copy())

                #if len(result[i]) == 0:
                result[i].extend(res_new)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        pass


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

    def repair(self,triplet):
        """
        repair：
        1.妻子/丈夫是成对出现的
        2.串式搜索：作词 作曲 歌手 所示专辑 // 出版社 作者
        """ 
        if triplet['predicate'] == '丈夫':
            triplet_new = {
                'object': triplet['subject'],
                'predicate':'妻子',
                'subject':triplet['object']
            }
        elif triplet['predicate'] == '妻子':
            triplet_new = {
                'object': triplet['subject'],
                'predicate':'丈夫',
                'subject':triplet['object']
            }    
        
               


