import os
import json
import time
import argparse

import torch

from typing import Dict, List, Tuple, Set, Optional

from prefetch_generator import BackgroundGenerator
from tqdm import tqdm

from torch.optim import Adam, SGD
from transformers import AdamW, WarmupLinearSchedule

from lib.preprocessings import Chinese_bert_preprocessing, Chinese_selection_preprocessing, Chinese_bert_pso_preprocessing
from lib.dataloaders import Selection_Dataset, Selection_loader
from lib.metrics import F1_triplet, F1_ner, F1_P, SaveError,SaveRecord,roc_auc_class,attn_plot,attn_multi_plot,attn_pso_plot,attn_pso_plot_sub,attn_pso_plot_stack
from lib.models import MultiHeadSelection
from lib.models.pso_model_new import P_Model_new,SO_Model_New
from lib.config import Hyper
from lib.kg.spo_search import SPO_searcher


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    '-e',
                    type=str,
                    default='conll_bert_re',
                    help='experiments/exp_name.json')
parser.add_argument('--mode',
                    '-m',
                    type=str,
                    default='preprocessing',
                    help='preprocessing|train|evaluation|reload|postcheck')
parser.add_argument('--epoch',
                    '-p',
                    type = int,
                    help = 'reload epoch')
parser.add_argument('--type',
                    '-t',
                    type = str,
                    default = 'selection',
                    help = 'selection|pso_1|pso_2')

args = parser.parse_args()


class Runner(object):
    def __init__(self, exp_name: str,model_name):
        self.exp_name = exp_name
        self.model_dir = 'saved_models'
        self.model_name = model_name
        self.hyper = Hyper(os.path.join('experiments',
                                        self.exp_name + '.json'))

        self.hyper.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu = self.hyper.gpu
        self.preprocessor = None
        self.triplet_metrics = F1_triplet()
        self.ner_metrics = F1_ner()
        self.save_err = SaveError()
        self.save_rc = SaveRecord(self.exp_name)
        self.p_metrics = F1_P()
        self.optimizer = None
        self.model = None
        self.model_p = None
        

    def _optimizer(self, name, model):
        no_decay = ["bias","LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

        m = {
            'adam': Adam(model.parameters()),
            'sgd': SGD(model.parameters(), lr=0.5),
            'adamw': AdamW(model.parameters(),lr = 2e-5),
            'adamw_pre': AdamW(optimizer_grouped_parameters, lr = self.hyper.lr ,eps = 1e-8)
        }
        return m[name]
    
    def _scheduler(self,optimizer,warmup_steps, t_total):
        return WarmupLinearSchedule(optimizer, warmup_steps, t_total)


    def _init_model(self,is_xgb = False):
        #self.model = MultiHeadSelection(self.hyper).cuda(self.gpu)
        if self.model_name == 'selection':
            self.model = MultiHeadSelection(self.hyper,is_xgb).to(self.hyper.device)
        elif self.model_name == 'pso_1':
            #self.model = P_Model(self.hyper).to(self.hyper.device)
            self.model = P_Model_new(self.hyper,is_xgb).to(self.hyper.device)
        elif self.model_name == 'pso_2':
            #self.model = SO_TAG_Model(self.hyper).to(self.hyper.device)
            #self.model = SO_WITHOUT_Model(self.hyper).to(self.hyper.device)
            self.model = SO_Model_New(self.hyper,is_xgb).to(self.hyper.device)
            self.model_p = P_Model_new(self.hyper,is_xgb).to(self.hyper.device)

    def preprocessing(self):
        if self.exp_name == 'chinese_selection_re':
            self.preprocessor = Chinese_selection_preprocessing(self.hyper)
        elif self.exp_name == 'chinese_bert_re':
            self.preprocessor = Chinese_bert_preprocessing(self.hyper)
        elif self.exp_name == 'chinese_bert_pso':
            self.preprocessor = Chinese_bert_pso_preprocessing(self.hyper)
        self.preprocessor.gen_relation_vocab()
        self.preprocessor.gen_all_data()
        self.preprocessor.gen_vocab(min_freq=1)
        self.preprocessor.gen_pso_vocab()
        self.preprocessor.gen_postag_vocab()
        self.preprocessor.gen_tag_vocab()
        # for ner only
        self.preprocessor.gen_bio_vocab()

    def run(self, mode: str):
        if mode == 'preprocessing':
            self.preprocessing()
        elif mode == 'train':
            self._init_model()
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
            if self.model_name in ('pso_1','pso_2'):
                self.train_pso()
            self.train()
        elif mode == 'evaluation':
            self._init_model()
            self.load_model(epoch=self.hyper.evaluation_epoch)
            if self.model_name == 'selection':
                self.evaluation()
            elif self.model_name in ('pso_1','pso_2'):
                self.evaluation_pso()
        elif mode == 'reload':
            self._init_model()
            self.load_model(epoch = args.epoch)
            self.optimizer = self._optimizer(self.hyper.optimizer, self.model)
            self.train(epoch=args.epoch+1)
        elif mode == 'postcheck':
            self._init_model(is_xgb= True)
            self.load_model(epoch=self.hyper.evaluation_epoch)
            if self.model_name in ('pso_1','pso_2'):
                self.postcheck_pso()
            else:
                self.postcheck()
        else:
            raise ValueError('invalid mode')

    def load_model(self, epoch: int):
        exp_name = self.exp_name

        if self.model_p:
            self.model_p.load_state_dict(
                torch.load(
                    os.path.join(self.model_dir,
                                exp_name + '_' + str(6)),
                                map_location=self.hyper.device))    


        if self.model_name == 'pso_2':
            exp_name = self.exp_name + self.model_name
        self.model.load_state_dict(
            torch.load(
                os.path.join(self.model_dir,
                            exp_name + '_' + str(epoch)),
                            map_location=self.hyper.device))   

    def save_model(self, epoch: int):
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        exp_name = self.exp_name
        if self.model_name == 'pso_2':
            exp_name = self.exp_name + self.model_name
        torch.save(
            self.model.state_dict(),
            os.path.join(self.model_dir, exp_name + '_' + str(epoch)))

    def postcheck(self):
        dev_set = Selection_Dataset(self.hyper,self.hyper.xgb_train_root,is_xgb=True)
        loader = Selection_loader(dev_set, batch_size=self.hyper.eval_batch, pin_memory=True)
        self.model.eval()
        
        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        self.save_rc.reset('pos_neg_score.json')

        flag = False
        
        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)
                if output.get('pos_list'):
                    pos_list = output['pos_list']
                    flag = True
                neg_list = output['neg_list']

                if flag:
                    self.save_rc.save(neg_list,pos_list)
                else:
                    self.save_rc.save(neg_list)

    def postcheck_pso(self):
        dev_set = Selection_Dataset(self.hyper,self.hyper.xgb_train_root,is_xgb= True)
        loader = Selection_loader(dev_set, batch_size=self.hyper.eval_batch, pin_memory=True)
        self.model.eval()

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        self.save_rc.reset('pos_neg_score.json')

        flag = False

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model_p(sample,is_train = False)
                if output.get('pos_score'):
                    pos_score = output['pos_score']
                    flag = True
                neg_score = output['neg_score']
                
                if flag:
                    dev_set.postcheck_schema_transformer(sample,neg_score,False)
                    output = self.model(sample,is_train = False)
                    neg_list = output['neg_list']
                    dev_set.postcheck_schema_transformer(sample,pos_score,True)
                    output = self.model(sample,is_train = False)
                    pos_list = output['pos_list']
                else:
                    dev_set.postcheck_schema_transformer(sample,neg_score,False)
                    output = self.model(sample,is_train = False)
                    neg_list = output['neg_list']

                if flag:
                    self.save_rc.save(neg_list,pos_list)
                else:
                    self.save_rc.save(neg_list)

    def evaluation_pso(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        loader = Selection_loader(dev_set, batch_size=self.hyper.eval_batch, pin_memory=True)
        self.p_metrics.reset()
        self.triplet_metrics.reset()
        self.model.eval()
        if self.model_name == 'pso_2':
            self.model_p.eval()
        all_labels,all_logits = [],[]
        self.model._init_spo_search(SPO_searcher(self.hyper))

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        if self.model_name == 'pso_1':
            with torch.no_grad():
                for batch_ndx, sample in pbar:
                    output =  self.model(sample,is_train = False)

                    """
                    if batch_ndx == 44:
                        attn = self.model.attn.cpu()
                        print(attn)
                        attn_plot(attn,sample.text[0])
                    """
                    
                    self.p_metrics(output['p_decode'],output['p_golden'])
                    all_labels.extend(output['p_golden'])
                    all_logits.extend(output['p_logits'])

                """
                with open('labels.txt','w',encoding = 'utf-8') as f:
                    for l in all_labels:
                        f.write(' '.join(l))
                        f.write('\n')
                
                with open('scores.txt','w',encoding ='utf-8') as t:
                    for l in all_logits:
                        t.write(' '.join([str(i) for i in l]))
                        t.write('\n')
                """

                p_result = self.p_metrics.get_metric()

                print('P-> ' +  ', '.join([
                    "%s: %.4f" % (name[0], value)
                    for name, value in p_result.items() if not name.startswith("_")
                ]))

                #roc_auc_class(all_labels,all_logits)

        elif self.model_name == 'pso_2':

            self.save_rc.reset('dev.json')

            with torch.no_grad():
                for batch_ndx, sample in pbar:
                    output_p =  self.model_p(sample,is_train = False)
                    self.p_metrics(output_p['p_decode'],output_p['p_golden'])
                    all_labels.extend(output_p['p_decode'])
                    all_logits.extend(output_p['p_golden'])

                    dev_set.schema_transformer(output_p,sample)
                    output = self.model(sample,is_train = False)

                    """
                    if batch_ndx == 63:
                        print(sample.text[0])
                        
                        attn = self.model.attn[2].cpu()
                        attn_pso_plot_sub(attn,sample.text[0])

                        for id in range(9,12):
                            attn = self.model.attn[id-9].cpu()
                            attn_pso_plot(attn,sample.text[0],id)
                            #attn_pso_plot_stack(attn,sample.text[0],id)
                        
                        attn = self.model.attn[-1].cpu()
                        attn_pso_plot_sub(attn,sample.text[0])
                        """
                    self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                    self.save_err.save(batch_ndx,output['selection_triplets'], output['spo_gold'])
                    self.save_rc.save(output['selection_triplets'])

                p_result = self.p_metrics.get_metric()
                triplet_result = self.triplet_metrics.get_metric()

                print('Triplets-> ' +  ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in triplet_result.items() if not name.startswith("_")]))

                print('P-> ' +  ', '.join([
                    "%s: %.4f" % (name[0], value)
                    for name, value in p_result.items() if not name.startswith("_")
                ]))

                #self.save_err.write(self.exp_name)


    def evaluation(self):
        dev_set = Selection_Dataset(self.hyper, self.hyper.dev)
        loader = Selection_loader(dev_set, batch_size=self.hyper.eval_batch, pin_memory=True)
        self.triplet_metrics.reset()
        self.model.eval()
        self.model._init_spo_search(SPO_searcher(self.hyper))

        pbar = tqdm(enumerate(BackgroundGenerator(loader)), total=len(loader))

        self.save_rc.reset('dev.json')

        with torch.no_grad():
            for batch_ndx, sample in pbar:
                output = self.model(sample, is_train=False)

                """
                attn = self.model.attn.cpu()
                if batch_ndx == 41:
                    print(sample.text[0])
                    attn_multi_plot(attn,sample.text[0])
                """

                # distant search
                self.triplet_metrics(output['selection_triplets'], output['spo_gold'])
                self.ner_metrics(output['gold_tags'], output['decoded_tag'])
                #self.save_err.save(batch_ndx,output['selection_triplets'], output['spo_gold'])
                self.save_rc.save(output['selection_triplets'])

            triplet_result = self.triplet_metrics.get_metric()
            ner_result = self.ner_metrics.get_metric()
            print('Triplets-> ' +  ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in triplet_result.items() if not name.startswith("_")
            ]) + ' ||' + 'NER->' + ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in ner_result.items() if not name.startswith("_")
            ]))
            #self.triplet_metrics.get_df()

        #self.save_err.write(self.exp_name)

    def train(self,epoch = None):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        #sampler = torch.utils.data.sampler.WeightedRandomSampler(train_set.weight,len(train_set.weight))
        loader = Selection_loader(train_set, batch_size=self.hyper.train_batch, pin_memory=True)
        #loader = Selection_loader(train_set, batch_size=self.hyper.train_batch,sampler = sampler, pin_memory=True)

        if not epoch:
            epoch = 0

        while epoch <= self.hyper.epoch_num:
        #for epoch in range(self.hyper.epoch_num):
            self.model.train()
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:

                self.optimizer.zero_grad()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            self.save_model(epoch)

            """
            if epoch % self.hyper.print_epoch == 0 and epoch > 3:
                if self.model_name == 'selection':
                    self.evaluation()
                elif self.model_name == 'pso_1':
                    self.evaluation_pso()
            """
            epoch += 1

    def train_pso(self, epoch = None):
        train_set = Selection_Dataset(self.hyper, self.hyper.train)
        loader = Selection_loader(train_set, batch_size=self.hyper.train_batch, pin_memory=True)

        if not epoch:
            epoch = 0

        num_train_steps = int(len(loader)/self.hyper.train_batch*(self.hyper.epoch_num-epoch + 1))
        num_warmup_steps = int(num_train_steps*self.hyper.warmup_prop)

        self.scheduler = self._scheduler(self.optimizer,num_warmup_steps,num_train_steps)

        while epoch <= self.hyper.epoch_num:
            pbar = tqdm(enumerate(BackgroundGenerator(loader)),
                        total=len(loader))

            for batch_idx, sample in pbar:
                self.model.train()
                output = self.model(sample, is_train=True)
                loss = output['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)

                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

                pbar.set_description(output['description'](
                    epoch, self.hyper.epoch_num))

            self.save_model(epoch)
            epoch += 1

if __name__ == "__main__":
    config = Runner(exp_name=args.exp_name,model_name = args.type)
    config.run(mode=args.mode)
