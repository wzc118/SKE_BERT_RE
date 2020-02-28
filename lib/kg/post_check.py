import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
import pickle
import math
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import shap

matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 

"""
triplet classification
1. get rank
2. get transe
3. SDValiate
"""

class XGB(object):
    def __init__(self,model_name = 'xgb_pso.model',nthread = 5, eval_metric = "error"):
        self.nthread = nthread
        self.model_name = model_name
        self.eval_metric = eval_metric
        self.init_flag = False
        self.xgb_root = "saved_models"
    
    def trainModel(self,train_x,train_y):
        self.clf = xgb.XGBClassifier(nthread= self.nthread)

        self.clf.fit(train_x,train_y,eval_metric= self.eval_metric,
                        eval_set = [(train_x,train_y)])

        self.init_flag = True

        evals_result = self.clf.evals_result()

        plt.style.use('ggplot') 
        xgb.plot_importance(self.clf, grid = True,importance_type='gain')
        plt.title('Feature importance (gain)')
        #plt.savefig('xgb_gain.png')

        print('eval_result',evals_result)
        
        #self.shap_plot(train_x)

        pickle.dump(self.clf,open(os.path.join(self.xgb_root,self.model_name),'wb'),True)

    def shap_plot(self,train_x):
        explainer = shap.TreeExplainer(self.clf)
        shap_values = explainer.shap_values(train_x)
        shap.summary_plot(shap_values, train_x, plot_type="bar",show=False)
        plt.savefig('shap_mean.pdf',format = 'pdf', dpi = 1000,bbox_inches = 'tight')

    def loadModel(self):
        try:
            self.clf = pickle.load(open(os.path.join(self.xgb_root,self.model_name),'rb'))
            self.init_flag = True
        except Exception:
            print("load model error")

    def predict(self,test_x):
        if not self.init_flag:
            self.loadModel()

        #pred_y = self.clf.predict(test_x)
        probs = self.clf.predict_proba(test_x)
        threshold = 0.85
        pred_y = probs[:,1]
        pred_y[pred_y >= threshold] = 1
        pred_y[pred_y < threshold] = 0
        
        return pred_y

class PostCheck(object):
    def __init__(self):
        #self.data_root = 'data/chinese_bert/multi_head_selection'
        self.data_root = 'data/chinese_bert/pso'
        self.op_cnt = Counter()
        self.o_cnt = Counter()
        self.so_cnt = Counter()
        self.p_schema = dict()
        self.transe_root = 'lib/transe'
        #self.record_root = 'error/chinese_bert_re'
        self.record_root = 'error/chinese_bert_pso'

        self.relation_vocab = json.load(
            open(os.path.join(self.data_root, 'relation_vocab.json'), 'r',encoding='utf-8'))

        self.model = XGB()

    def build_kg(self):
        for line in open(os.path.join('raw_data','chinese','all_50_schemas'),'r',encoding = 'utf-8'):
            line = line.strip("\n")
            instance = json.loads(line)
            o_type,p,s_type = instance['object_type'],instance['predicate'],instance['subject_type']
            self.p_schema[p] = [o_type,s_type]

        for line in open(os.path.join(self.data_root, 'train_data.json'), 'r',encoding='utf-8'):
            line = line.strip("\n")
            instance = json.loads(line)
            spo_list = instance['spo_list']

            for spo in spo_list:
                s = ''.join(spo['subject'])
                o = ''.join(spo['object'])
                p = spo['predicate']

                self.o_cnt[o] += 1
                self.op_cnt[(p,o)] += 1
                self.so_cnt[(s,o)] += 1


    def build_transe(self):
        # ent2vec
        with open(os.path.join(self.transe_root,'train_ent_emb_pso.pickle'), 'rb') as handle:
            ent2vec = pickle.load(handle)
        with open(os.path.join(self.transe_root,'train_rel_emb_pso.pickle'), 'rb') as handle:
            rel2vec = pickle.load(handle)

        thre_dict, trans_dict = {},{}

        def getThreshold(rrank):
            distanceFlagList = rrank
            distanceFlagList = sorted(distanceFlagList, key = lambda sp:sp[0], reverse = False)
            
            threshold = distanceFlagList[0][0] - 0.01
            maxValue = 0
            currenrValue = 0
            for i in range(1,len(distanceFlagList)):
                if distanceFlagList[i-1][1] == 1:
                    currenrValue += 1
                else:
                    currenrValue -= 1

                if currenrValue > maxValue:
                    threshold = (distanceFlagList[i][0] + distanceFlagList[i-1][0]) / 2.0
                    maxValue = currenrValue
            return threshold

        for line in open(os.path.join(self.record_root ,'pos_neg_score.json'),'r',encoding = 'utf-8'):
            line = line.strip("\n")
            tmp = json.loads(line)
            for spo in tmp['pos']:
                o,s,p,label = spo['object'],spo['subject'],spo['predicate'],1
                o,s = ''.join(o.split(' ')),''.join(s.split(' '))
                try:
                # get threshold
                    ss = ent2vec[s] + rel2vec[p] - ent2vec[o] 
                except:
                    print(s,p,o)
                    continue
                transV = np.linalg.norm(ss, ord = 2)
                if p not in trans_dict.keys():
                    trans_dict[p] = [(transV,int(label))]
                else:
                    trans_dict[p].append((transV,int(label)))

            for spo in tmp['neg']:
                o,s,p,label = spo['object'],spo['subject'],spo['predicate'],0
                o,s = ''.join(o.split(' ')),''.join(s.split(' '))
                try:
                    ss = ent2vec[s] + rel2vec[p] - ent2vec[o] 
                except:
                    print(s,p,o)
                    continue
                transV = np.linalg.norm(ss, ord = 2)
                if p not in trans_dict.keys():
                    trans_dict[p] = [(transV,int(label))]
                else:
                    trans_dict[p].append((transV,int(label)))
        
        for it in trans_dict.keys():
            thre_dict[it] = getThreshold(trans_dict[it])

        print(thre_dict)
        
        return thre_dict,ent2vec,rel2vec

       
    def check_rdf(self,o,p,s):
        # smaller than significance 0.05 but bigger than 0
        if self.o_cnt[o] != 0:
            return self.op_cnt[(p,o)] / self.o_cnt[o]
        else:
            # a new word not in kg
            return 1

    def transe_score(self,o,p,s):
        threshold = self.thre_dict[p]
        o,s = ''.join(o.split(' ')),''.join(s.split(' '))
        try:
            ss = self.ent2vec[s] + self.rel2vec[p] - self.ent2vec[o]
        except:
            print(s,p,o)
            return
        transV = np.linalg.norm(ss, ord = 2)
        f = 1.0 / (1.0 + math.exp(-10 * (threshold - transV)))
        return f 

    def train(self):
        self.build_kg()
        self.thre_dict,self.ent2vec,self.rel2vec = self.build_transe()

        data = []

        idx = 0
        for line in open(os.path.join(self.record_root ,'pos_neg_score.json'),'r',encoding = 'utf-8'):
            line = line.strip("\n")
            tmp = json.loads(line)
            for spo in tmp['pos']:
                s,p,o,label,score,seg = spo['subject'],spo['predicate'],spo['object'],1,spo['score'],spo['seg']
                distant = 1 if self.so_cnt[(s,o)] > 1 else 0
                format_ = {'object':o,'predicate':p,'subject':s,'label':label,'score':score,'distant':distant,'seg':seg,'num':idx}
                data.append(format_)
            for spo in tmp['neg']:
                s,p,o,label,score,seg = spo['subject'],spo['predicate'],spo['object'],0,spo['score'],spo['seg']
                distant = 1 if self.so_cnt[(s,o)] > 1 else 0
                format_ = {'object':o,'predicate':p,'subject':s,'label':label,'score':score,'distant':distant,'seg':seg,'num':idx}
                data.append(format_)
            idx += 1
        
        data = pd.DataFrame.from_records(data)

        # sdvaliate
        data['sdvaliate'] = data.apply(lambda x: self.check_rdf(x.object,x.predicate,x.subject),axis = 1)
        # transe 
        data['transe'] = data.apply(lambda x: self.transe_score(x.object, x.predicate, x.subject), axis = 1)
        # rank
        data['rank'] = data["score"].groupby(data["num"]).rank(pct = True,ascending = False)
        # all records
        g = data.groupby('num')['predicate'].apply(set).map(list)
        def getvalue(x):
            return g[x]

        data['list'] = data['num'].map(getvalue)

        data = data.drop(['list','num','object','predicate','subject'],1).join(
            pd.get_dummies(pd.DataFrame(data.list.tolist()).stack()).astype(int).sum(level = 0) \
            .reindex(columns = list(self.relation_vocab.keys())).drop(['N'],axis = 1)
        )

        # change position
        data_ = data.pop("label")
        data["label"] = data_

        X = data.iloc[:,:-1]
        #X = data.iloc[:,:-1].as_matrix()
        y = data['label']

        self.model.trainModel(X,y)


    def test(self):
        self.build_kg()
        self.thre_dict,_,_ = self.build_transe()

        # ent2vec
        with open(os.path.join(self.transe_root,'dev_ent_emb_pso.pickle'), 'rb') as handle:
            self.ent2vec = pickle.load(handle)
        with open(os.path.join(self.transe_root,'dev_rel_emb_pso.pickle'), 'rb') as handle:
            self.rel2vec = pickle.load(handle)

        data = []

        idx = 0
        for line in open(os.path.join(self.record_root,'dev.json'),'r',encoding = 'utf-8'):
            tmp = json.loads(line)
            try:
                for spo in tmp['neg']:
                    s,p,o,score,seg = spo['subject'],spo['predicate'],spo['object'],spo['score'],spo['seg']
                    distant = spo['distant']
                    format_ = {'object':o,'predicate':p,'subject':s,'score':score,'distant':distant,'seg':seg,'num':idx}
                    data.append(format_)
            except:
                    format_ = {'object':None,'predicate':None,'subject':None,'score':None,'distant':None,'seg':None,'num':idx}
                    data.append(format_)
        
            idx += 1
        
        data = pd.DataFrame.from_records(data)
        data = data.dropna()

        print(len(data.loc[data['distant'] == 0].drop_duplicates()))

        # sdvaliate
        data['sdvaliate'] = data.apply(lambda x: self.check_rdf(x.object,x.predicate,x.subject),axis = 1)
        # transe 
        data['transe'] = data.apply(lambda x: self.transe_score(x.object, x.predicate, x.subject), axis = 1)
        # rank
        data['rank'] = data["score"].groupby(data["num"]).rank(pct = True,ascending = False)
        # all records
        g = data.groupby('num')['predicate'].apply(set).map(list)
        def getvalue(x):
            return g[x]
        data['list'] = data['num'].map(getvalue)
        #print(data.head(50))
        data_df = data.drop(['list','num','object','predicate','subject'],1).join(
            pd.get_dummies(pd.DataFrame(data.list.tolist()).stack()).astype(int).sum(level = 0) \
            .reindex(columns = list(self.relation_vocab.keys())).drop(['N'],axis = 1)
        )

        X = data_df
        #X = data_df.as_matrix()
        y_pred = self.model.predict(X)

        # join
        data['predict'] = y_pred
        data = data.loc[data['predict'] == 1]
        data['spo'] = data.apply(lambda x: '%s_%s_%s' %(x['subject'],x['predicate'],x['object']),axis = 1)
        ans  = data[['num','spo']]
        ans_l = ans.groupby(ans.num)['spo'].agg({'size_l': len, 'set_l': lambda x: set(x)}).reset_index()
        ans_l['size_l'] = ans_l['set_l'].apply(len)
        print(ans_l.head(-1))
        
        index = 0
        res = []
        with open(os.path.join(self.data_root,'dev_data_distillation.json'), 'r',encoding='utf-8') as f:
            for line in f:
                tmp = json.loads(line)
                for spo in tmp['spo_list']:
                    s,p,o =  ''.join(spo['subject']),spo['predicate'],''.join(spo['object'])
                    s_p_o = s + '_' + p + '_' + o
                    format_ = {'num':index, 'spo':s_p_o}
                    res.append(format_)
                index += 1
        
        ans_r = pd.DataFrame.from_records(res)
        ans_r = ans_r.groupby(ans_r.num)['spo'].agg({'size_r': len, 'set_r': lambda x: set(x)}).reset_index()
        ans_r['size_r'] = ans_r['set_r'].apply(len)
        print(ans_r.head(-1))

        ans = pd.merge(ans_l,ans_r, on = "num",how = "right")
        #ans = ans.fillna(0) 
        ans['size_l'] = ans['size_l'].replace(np.nan, 0)
        ans['set_l'] = ans['set_l'].apply(lambda x: x if isinstance(x,set) else set())
        #print(ans[pd.isnull(ans).any(axis=1)])
        #ans =  ans_l.merge(ans_r, left_on = "num", right_on = "num")
        ans["match"] = [(ans.loc[r,"set_l"]) & (ans.loc[r,"set_r"]) for r in range(len(ans))]
        ans["size_match"] = ans["match"].str.len()

        ans = ans.sort_values(by=['num'])
        print(ans.head(-1))

        with open(os.path.join(self.record_root ,'dev_ensemble.json'),'w',encoding = 'utf-8') as t:
            for index, row in ans.iterrows():
                instance,neg = {},[]
                num,tmp = row['num'],list(row['set_l'])
                if len(tmp):
                    for spo in tmp:
                        try:
                            [s,p,o] = spo.split('_')
                        except:
                            [s,p,o] = spo.split('_')[:3]
                            print(spo)
                        neg.append({'subject':s,'predicate':p,'object':o})
                else:
                    pass
                instance['neg'] = neg
                ins_json = json.dumps(instance, ensure_ascii=False)
                t.write(ins_json)
                t.write('\n')

        #ans.to_csv('error/chinese_bert_re/post_check_join.csv')

        # precision
        inter = sum(ans["size_match"].values)
        left = sum(ans["size_l"].values)
        right = sum(ans["size_r"].values)
        
        f1, p, r = 2 * inter / (left + right), inter / left, inter / right
        result = {"precision": p, "recall": r, "fscore": f1}
        print('Triplets-> ' +  ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in result.items() if not name.startswith("_")]))
    
        def func(x):
            if x == 1:
                return str(x)
            elif 2 <=x <= 5:
                return '2到5'
            elif 6 <=x <= 10:
                return '6到10'
            else:
                return '大于10'

        ans['size_rr'] = ans['size_r'].apply(func)
        new_df = ans.groupby(['size_rr'])[['size_l','size_r','size_match']].sum().reset_index()
        new_df['f1'] = 2*new_df['size_match']/(new_df['size_l']+new_df['size_r'])
        print(new_df)

    def test_pso(self):
        self.build_kg()
        self.thre_dict,_,_ = self.build_transe()

        # ent2vec
        with open(os.path.join(self.transe_root,'dev_ent_emb_pso.pickle'), 'rb') as handle:
            self.ent2vec = pickle.load(handle)
        with open(os.path.join(self.transe_root,'dev_rel_emb_pso.pickle'), 'rb') as handle:
            self.rel2vec = pickle.load(handle)

        data = []

        idx = 0
        for line in open(os.path.join(self.record_root,'dev.json'),'r',encoding = 'utf-8'):
            tmp = json.loads(line)
            try:
                for spo in tmp['neg']:
                    s,p,o,score,seg = spo['subject'],spo['predicate'],spo['object'],spo['score'],spo['seg']
                    distant = spo['distant']
                    format_ = {'object':o,'predicate':p,'subject':s,'score':score,'distant':distant,'seg':seg,'num':idx}
                    data.append(format_)
            except:
                    format_ = {'object':None,'predicate':None,'subject':None,'score':None,'distant':None,'seg':None,'num':idx}
                    data.append(format_)
        
            idx += 1
        
        data = pd.DataFrame.from_records(data)
        data = data.dropna()

        print(len(data.loc[data['distant'] == 0].drop_duplicates()))

        # sdvaliate
        data['sdvaliate'] = data.apply(lambda x: self.check_rdf(x.object,x.predicate,x.subject),axis = 1)
        # transe 
        data['transe'] = data.apply(lambda x: self.transe_score(x.object, x.predicate, x.subject), axis = 1)
        # rank
        data['rank'] = data["score"].groupby(data["num"]).rank(pct = True,ascending = False)
        # all records
        g = data.groupby('num')['predicate'].apply(set).map(list)
        def getvalue(x):
            return g[x]
        data['list'] = data['num'].map(getvalue)
        #print(data.head(50))
        data_df = data.drop(['list','num','object','predicate','subject'],1).join(
            pd.get_dummies(pd.DataFrame(data.list.tolist()).stack()).astype(int).sum(level = 0) \
            .reindex(columns = list(self.relation_vocab.keys())).drop(['N'],axis = 1)
        )

        X = data_df
        #X = data_df.as_matrix()
        y_pred = self.model.predict(X)

        # join
        data['predict'] = y_pred
        data = data.loc[(data['distant'] == 0) | (data['predict'] == 1)]
        #data = data.loc[data['predict'] == 1]
        data['spo'] = data.apply(lambda x: '%s_%s_%s' %(x['subject'],x['predicate'],x['object']),axis = 1)
        ans  = data[['num','spo']]
        ans_l = ans.groupby(ans.num)['spo'].agg({'size_l': len, 'set_l': lambda x: set(x)}).reset_index()
        ans_l['size_l'] = ans_l['set_l'].apply(len)
        print(ans_l.head(-1))
        
        index = 0
        res = []
        with open(os.path.join(self.data_root,'dev_data_distillation.json'), 'r',encoding='utf-8') as f:
            for line in f:
                tmp = json.loads(line)
                for spo in tmp['spo_list']:
                    s,p,o =  ''.join(spo['subject']),spo['predicate'],''.join(spo['object'])
                    s_p_o = s + '_' + p + '_' + o
                    format_ = {'num':index, 'spo':s_p_o}
                    res.append(format_)
                index += 1
        
        ans_r = pd.DataFrame.from_records(res)
        ans_r = ans_r.groupby(ans_r.num)['spo'].agg({'size_r': len, 'set_r': lambda x: set(x)}).reset_index()
        ans_r['size_r'] = ans_r['set_r'].apply(len)
        print(ans_r.head(-1))

        ans = pd.merge(ans_l,ans_r, on = "num",how = "right")
        #ans = ans.fillna(0) 
        ans['size_l'] = ans['size_l'].replace(np.nan, 0)
        ans['set_l'] = ans['set_l'].apply(lambda x: x if isinstance(x,set) else set())
        #print(ans[pd.isnull(ans).any(axis=1)])
        #ans =  ans_l.merge(ans_r, left_on = "num", right_on = "num")
        ans["match"] = [(ans.loc[r,"set_l"]) & (ans.loc[r,"set_r"]) for r in range(len(ans))]
        ans["size_match"] = ans["match"].str.len()

        ans = ans.sort_values(by=['num'])
        print(ans.head(-1))

        with open(os.path.join(self.record_root ,'dev_ensemble.json'),'w',encoding = 'utf-8') as t:
            for index, row in ans.iterrows():
                instance,neg = {},[]
                num,tmp = row['num'],list(row['set_l'])
                if len(tmp):
                    for spo in tmp:
                        try:
                            [s,p,o] = spo.split('_')
                        except:
                            [s,p,o] = spo.split('_')[:3]
                            print(spo)
                        neg.append({'subject':s,'predicate':p,'object':o})
                else:
                    pass
                instance['neg'] = neg
                ins_json = json.dumps(instance, ensure_ascii=False)
                t.write(ins_json)
                t.write('\n')

        #ans.to_csv('error/chinese_bert_re/post_check_join.csv')

        # precision
        inter = sum(ans["size_match"].values)
        left = sum(ans["size_l"].values)
        right = sum(ans["size_r"].values)
        
        f1, p, r = 2 * inter / (left + right), inter / left, inter / right
        result = {"precision": p, "recall": r, "fscore": f1}
        print('Triplets-> ' +  ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in result.items() if not name.startswith("_")]))
    
        def func(x):
            if x == 1:
                return str(x)
            elif 2 <=x <= 5:
                return '2到5'
            elif 6 <=x <= 10:
                return '6到10'
            else:
                return '大于10'

        ans['size_rr'] = ans['size_r'].apply(func)
        new_df = ans.groupby(['size_rr'])[['size_l','size_r','size_match']].sum().reset_index()
        new_df['f1'] = 2*new_df['size_match']/(new_df['size_l']+new_df['size_r'])
        print(new_df)


    def statistic(self):
        index = 0
        res = []
        with open(os.path.join(self.data_root,'dev_data_distillation.json'), 'r',encoding='utf-8') as f:
            for line in f:
                tmp = json.loads(line)
                for spo in tmp['spo_list']:
                    s,p,o =  ''.join(spo['subject']),spo['predicate'],''.join(spo['object'])
                    s_p_o = s + '_' + p + '_' + o
                    format_ = {'num':index, 'spo':s_p_o}
                    res.append(format_)
                index += 1

        index = 0
        data = []
        for line in open(os.path.join('error/chinese_bert_pso','bert_dev_pso_distillation.json'),'r',encoding = 'utf-8'):
            tmp = json.loads(line)
            for spo in tmp['neg']:
                s,p,o = spo['subject'],spo['predicate'],spo['object']
                s_p_o = s + '_' + p + '_' + o
                format_ = {'num':index,'spo':s_p_o}
                data.append(format_)
            index += 1

        df_r = pd.DataFrame.from_records(res).dropna()
        df_r = df_r.groupby(df_r.num)['spo'].agg({'size_r':len,'set_r':lambda x:set(x)}).reset_index()
        df_r['size_r'] = df_r['set_r'].apply(len)
        df_l = pd.DataFrame.from_records(data).dropna()
        df_l = df_l.groupby(df_l.num)['spo'].agg({'size_l':len,'set_l':lambda x:set(x)}).reset_index()
        df_l['size_l'] = df_l['set_l'].apply(len)

        ans = pd.merge(df_l,df_r,on = 'num',how ='right')
        ans['size_l'] = ans['size_l'].replace(np.nan, 0)
        ans['set_l'] = ans['set_l'].apply(lambda x: x if isinstance(x,set) else set())
        ans["match"] = [(ans.loc[r,"set_l"]) & (ans.loc[r,"set_r"]) for r in range(len(ans))]
        ans["size_match"] = ans["match"].str.len()

        # calc_pr
        inter = sum(ans["size_match"].values)
        left = sum(ans["size_l"].values)
        right = sum(ans["size_r"].values)

        f1, p, r = 2 * inter / (left + right), inter / left, inter / right
        result = {"precision": p, "recall": r, "fscore": f1}
        print('Triplets-> ' +  ', '.join([
                "%s: %.4f" % (name[0], value)
                for name, value in result.items() if not name.startswith("_")]))
    
        def func(x):
            if x == 1:
                return str(x)
            elif 2 <=x <= 5:
                return '2到5'
            elif 6 <=x <= 10:
                return '6到10'
            else:
                return '大于10'

        ans['size_rr'] = ans['size_r'].apply(func)
        new_df = ans.groupby(['size_rr'])[['size_l','size_r','size_match']].sum().reset_index()
        new_df['f1'] = 2*new_df['size_match']/(new_df['size_l']+new_df['size_r'])
        print(new_df)
        

if __name__ == "__main__":
    tester = PostCheck()
    tester.test_pso()



        









