# -*- coding:utf-8 -*-
import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties
from get_data import get_data

# font = FontProperties(fname='/Library/Fonts/Songti.ttc')


def draw():
    # 定义热图的横纵坐标
    xLabel = []
    yLabel = []

    levels_base = {
        '1': 8,
        '2': 28,
        '3': 56,
        '4': 70,
        '5': 56,
        '6': 28,
        '7': 8,
        '8': 1,
    }

    levels_base = {
        '1': 8,
        '2': 2,
        '3': 2,
        '4': 2,
        '5': 2
    }

    levels = {
        '1': 8,
        '2': 4,
        '3': 3,
        '4': 2,
    }

    a1, a2 = 0, 0

    for level_base_label in levels_base:
        if list(levels.values())[0] != levels_base[level_base_label]:
            a1 = a1 + levels_base[level_base_label]
        else:
            break

    for level_base_label in levels_base:
        a2 = a2 + levels_base[level_base_label]
        if list(levels.values())[-1] == levels_base[level_base_label]:
            break

    a1, a2 = 0, 17
    for level_label in levels:
        if len(levels) > 1:
            level_number = levels[level_label]
            is_label = True
            for tuple_number in range(level_number):
              if is_label:
                  xLabel.append(level_label)
                  yLabel.append(level_label)
                  is_label = False
              else:
                  xLabel.append("")
                  yLabel.append("")
        else:
            level_number = levels[level_label]
            for tuple_number in range(level_number):
                xLabel.append(str(tuple_number + 1))
                yLabel.append(str(tuple_number + 1))

    # 准备数据阶段，利用random生成二维数据（5*5）
    data = []
    data = get_data()
    import torch
    data = torch.Tensor(data)
    data = data[a1:a2, a1:a2]
    data = data.numpy()
    import torch


    data = torch.Tensor(data)
    data = data.numpy()
    # 作图阶段
    fig = plt.figure(dpi = 300)
    # 定义画布为1*1个划分，并在第1个位置上进行作图
    ax = fig.add_subplot(111)
    # 定义横纵坐标的刻度
    ax.set_yticks(range(len(yLabel)), fontsize=5)
    ax.set_yticklabels(yLabel, fontproperties='simhei', fontsize=5)
    ax.set_xticks(range(len(xLabel)), fontsize=5)
    ax.set_xticklabels(xLabel, fontsize=5)
    # 作图并选择热图的颜色填充风格，这里选择hot
    im = ax.imshow(data, cmap=plt.cm.hot_r)
    # 增加右侧的颜色刻度条
    plt.colorbar(im)
    # 增加标题
    plt.title("Attention Score for Level 2 Feature", fontproperties='simhei')

    plt.savefig('/home/guofei/Multiple_Level_attention_score.jpg', dpi=300)
    plt.savefig('/home/guofei/Level_2_attention_score.jpg', dpi=300)
    # show
    plt.show()
d = draw()
