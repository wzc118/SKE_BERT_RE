import seaborn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties

#matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 

#matplotlib.get_cachedir()
#font = font_manager.FontProperties(fname="/root/wzc118/pytorch_multi_head_selection_re/simhei.ttf")
#seaborn.set(font=font.get_name())


def draw(data, x, y, ax):
    seaborn.heatmap(data, 
                    xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                    cbar=False, cmap="Greens",ax =ax)

def attn_plot(attn,text):
    sep_idx = text.index('[SEP]')
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    for h in range(4):
        draw(attn[0,h].data[:sep_idx+1,:sep_idx+1],text[:sep_idx],text[:sep_idx],ax = axs[h])
    plt.savefig('attn_plot.png')
    
def attn_multi_plot(attn,text):
    sep_idx = text.index('[SEP]')
    fig, axs = plt.subplots(1,4, figsize=(20, 10))
    for h in range(9,13):
        draw(attn[0,h].data[1:sep_idx+1,1:sep_idx+1],text[1:sep_idx],text[1:sep_idx],ax = axs[h])
    plt.savefig('attn_multi_plot.png')

def attn_pso_plot(attn,text,id):
    sep_idx = text.index('[SEP]')
    fig,axs = plt.subplots(1,4,figsize = (20,10))
    for h in range(8,12):
       draw(attn[0,h].data[:sep_idx,:sep_idx],text[:sep_idx],text[:sep_idx],ax = axs[h-8])
    plt.savefig('attn_pso_layer_{}_plot.png'.format(id)) 

def attn_pso_plot_sub(attn,text):
    sep_idx = text.index('[SEP]')
    fig,axs = plt.subplots(figsize = (4,4))
    #bottom, top = axs.get_ylim()
    draw(attn[0,7].data[:sep_idx,:sep_idx],text[:sep_idx],text[:sep_idx],ax = axs)
    #axs.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('attn_pso_layer11_head7.png') 

def attn_pso_plot_stack(attn,text,id):
    sep_idx = text.index('[SEP]')
    fig,axs = plt.subplots(1,1,figsize = (10,10))
    attn = attn.max(1)[0]
    draw(attn[0].data[:sep_idx,:sep_idx],text[:sep_idx],text[:sep_idx],ax = axs)
    plt.savefig('attn_pso_layer_{}_plot.png'.format(id))