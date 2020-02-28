import os
import json
from tqdm import tqdm
import pandas as pd

error = json.load(open('error/chinese_bert_re/comp_score.json','r',encoding = 'utf-8'))
pso_schema = json.load(open('data/chinese_bert/multi_head_selection/pso_schema.json','r',encoding = 'utf-8'))

def _check(tmp,gold):
    s,p,o = tmp['subject'],tmp['predicate'],tmp['object']
    return any((s == g['subject'] and p == g['predicate'] and o == g['object']) for g in gold)

def _check_mu(tmp,gold):
    s,p,o = tmp['subject'],tmp['predicate'],tmp['object']
    for g in gold:
        if p == g['predicate'] and len(g['subject']) > len(s) and g['subject'].endswith(s) and g['object'] == o:
            return True
        if p == g['predicate'] and len(g['object']) > len(o) and g['subject'].endswith(s) and g['subject'] == s:
            return True
    return False

cnt = 0

for e in error:
    predict = e['predict']
    gold = e['gold']
    # chenck score significance
    for tmp in predict:
        if (tmp['score'] >= 0.95 or (tmp['score'] >= 0.5 and tmp['distant'] == 1)) and not _check(tmp,gold):
            #print(tmp)
            cnt += 1

print(cnt)

data = []
with open('raw_data/chinese/dev_data.json','r',encoding = 'utf-8') as f:
    for l in tqdm(f):
        d = json.loads(l)
        data.append(d)
        
data_df = pd.DataFrame.from_records(data)
data_df["text"] = data_df["text"].map(lambda x: x.lower())
data_df.set_index("text")

#print(data_df.spo_list.values[:50])

cnt = 0
# 快速模糊查找
for e in error:
    predict = e['predict']
    gold = e['gold']
    # chenck score significance
    for tmp in predict:
        if (tmp['score'] >= 0.95 or (tmp['score'] >= 0.5 and tmp['distant'] == 1)) and not _check(tmp,gold):
            if tmp['predicate'] == '目' and _check_mu(tmp,gold):
                continue
            try:
                index = list(data_df[data_df.text.str.contains(tmp['subject'],regex=False) & data_df.text.str.contains(tmp['object'],regex=False)].index)
                if len(index) < 10:
                    for idx in index:
                        cnt += 1
                        data[idx]['spo_list'].append({"predicate": tmp['predicate'],"object_type":pso_schema[tmp['predicate']][1],
                           "subject_type":pso_schema[tmp['predicate']][0], \
                           "object":tmp['object'],"subject":tmp['subject']})
            except:
                continue
print(cnt)

with open('raw_data/chinese/dev_data_repair.json','w',encoding='utf-8') as t:
    for line in data:
        line = json.dumps(line,ensure_ascii=False)
        if line is not None:
            t.write(line)
            t.write('\n')





