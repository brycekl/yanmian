import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


res = np.zeros((7, 8))
var_list = [1, 2, 5, 10, 20, 40, 60, 80]
max_list = [1, 2, 4, 6, 8, 10, 12]
data_root = '../models/biye/var_max_test/'
for var_max_p in os.listdir(data_root):
    if os.path.isfile(os.path.join(data_root, var_max_p)): continue
    var_max = int(var_max_p.split('max')[-1])
    for var_p in os.listdir(os.path.join(data_root, var_max_p)):
        if os.path.isfile(os.path.join(data_root, var_max_p, var_p)): continue
        var = int(var_p.split('var')[-1].split('_')[0])
        mse = float(var_p.split('var')[-1].split('_')[-1])
        res[max_list.index(var_max), var_list.index(var)] = mse

# res = (res - res.min()) / (res.max() - res.min())
# res = 1 - res
# 归一化
# 找出大于5的值的索引
indices = res > 5
big_data = res[indices]
nor_res = np.zeros_like(res)
nor_res[indices] = (big_data - big_data.min()) / (big_data.max() - big_data.min()) + res[~indices].max()
nor_res[~indices] = res[~indices]
nor_res = (nor_res - nor_res.min()) / (nor_res.max() - nor_res.min())
# nor_res[max_list.index(8), var_list.index(40)] = 0.01
sns.heatmap(nor_res, cmap='hot')
# 绘制热力图
# plt.imshow(res, cmap='hot', interpolation='nearest')

# 添加颜色条
# plt.colorbar()

# 显示横轴与纵轴
plt.xlabel('Var')
plt.ylabel('Max_value')
plt.xticks(ticks=list(range(len(var_list))), labels=var_list)
plt.yticks(ticks=list(range(len(max_list))), labels=max_list)
plt.title('Normalized Mse')

# 显示图形
plt.savefig('test.png')
plt.show()

