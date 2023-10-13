# -*- coding:utf-8 -*-
import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
from matplotlib.font_manager import FontProperties

import numpy as np
import matplotlib.pyplot as plt
import string
HR_5 = np.array([[75.22, 76.34, 75.31, 78.03, 76.57],
                 [80.52, 82.93, 81.33, 83.97, 83.41],
                 [78.70, 80.41, 79.12, 82.91, 81.44],
                 [80.04, 82.66, 81.03, 83.87, 83.28],
                 [78.12, 79.26, 79.21, 80.14, 80.52]])


# 坐标轴-y （行标）
tag_y = ["$α$=1", "$α$=2", "$α$=3", "$α$=4", "$α$=5"]
# 坐标轴-x （列标）
tag_x = ["$β$=1", "$β$=2", "$β$=3", "$β$=4", "$β$=5"]
tag_y = []
# 坐标轴-x （列标）
tag_x = []

levels = {
    'level1': 8,
    'level2': 28,
    'level3': 56,
'level4': 70,
'level5': 56,
'level6': 28,
'level7': 8,
'level8': 1,
}

for level_label in levels:
    level_number =  levels[level_label]
    for tuple_number in range(level_number):
      tag_label = level_label + "_" + str(tuple_number)
      tag_x.append(tag_label)
      tag_y.append(tag_label)

fig, ax = plt.subplots()

ax.set_xticks(np.arange(len(tag_x)))
# 设置x轴刻度间隔，参数为x轴刻度长度，其实也可以写作np.arange(0, 5, 1)，目的就是提供5个刻度
ax.set_yticks(np.arange(len(tag_y)))
# 设置y轴刻度间隔
ax.set_xticklabels(tag_x)
# 设置x轴标签
ax.set_yticklabels(tag_y)

plt.imshow(HR_5, cmap='coolwarm', origin='upper', aspect="auto")


default_font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 14}
#plt.xlabel('Values of the parameter $β$\n(a) HR@5 (%)\n', default_font)
#plt.ylabel('Values of the parameter $α$', default_font)

plt.show()

plt.colorbar()
