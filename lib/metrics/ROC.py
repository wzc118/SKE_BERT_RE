from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp

fpr = dict()
tpr = dict()

roc_auc = dict()

map_ = {"祖籍": 0, "父亲": 1, "总部地点": 2, "出生地": 3, "目": 4, "面积": 5, "简称": 6, "上映时间": 7, "妻子": 8, "所属专辑": 9, "注册资本": 10, "首都": 11, "导演": 12, "字": 13, "身高": 14, "出品公司": 15, "修业年限": 16, "出生日期": 17, "制片人": 18, "母亲": 19, "编剧": 20, \
    "国籍": 21, "海拔": 22, "连载网站": 23, "丈夫": 24, "朝代": 25, "民族": 26, "号": 27, "出版社": 28, "主持人": 29, "专业代码": 30, "歌手": 31, "作词": 32, "主角": 33, "董事长": 34, "成立日期": 35, "毕业院校": 36, "占地面积": 37, "官方语言": 38, "邮政编码": 39, "人口数量": 40, "所在城市": 41, \
    "作者": 42, "作曲": 43, "气候": 44, "嘉宾": 45, "主演": 46, "改编自": 47, "创始人": 48}

one_hot = MultiLabelBinarizer(classes= list(map_.keys()))

def one_hot_(x):
    return one_hot.fit_transform(x)

def roc_auc_class(all_labels,all_logits):
    all_labels = one_hot_(all_labels)
    all_logits = one_hot_(all_logits)
    labels = list(one_hot.classes_)

    for i in range(len(labels)):
        if i == 39:
            continue
        fpr[labels[i]],tpr[labels[i]],_ = roc_curve(all_labels[:,i],all_logits[:,i])
        roc_auc[labels[i]] = auc(fpr[labels[i]],tpr[labels[i]])

    fpr["micro"],tpr["micro"],_ = roc_curve(all_labels.ravel(),all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"],tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[labels[i]] for i in range(len(labels)) if i != 39]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(len(labels)):
        if i == 39:
            continue
        mean_tpr += interp(all_fpr,fpr[labels[i]],tpr[labels[i]])

    mean_tpr /= len(labels)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"],tpr["macro"])

    print(roc_auc)

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)


    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    lw = 2
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('p_plot.jpg')
    plt.show()