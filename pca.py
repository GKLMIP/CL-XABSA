import json

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # 加载PCA算法包
from sklearn.metrics import silhouette_score,calinski_harabasz_score
import numpy as np

def plot_circle(center=(3, 3),r=2):
    x = np.linspace(center[0] - r, center[0] + r, 5000)
    y1 = np.sqrt(r**2 - (x-center[0])**2) + center[1]
    y2 = -np.sqrt(r**2 - (x-center[0])**2) + center[1]
    plt.plot(x, y1, c='k')
    plt.plot(x, y2, c='k')

def creat_circle(center=(3, 3),r=2):

    circle=plt.Circle((center[0],center[1]),radius=r,fc='y',ec='r')#圆心坐标，半径，内部及边缘填充颜色

    return circle

def distance(x1,x2,y1,y2):
    return ((x1-x2) ** 2 + (y1-y2) ** 2) ** 0.5
# method = 'tl_embedding-5555'
# method = 'original_model_embedding'
method = 'acsmtl_model_embedding-5555'
with open(method+'/en_test_embedding.json') as f:
    x = json.load(f)
    y = [0] * len(x)

with open(method+'/fr_test_embedding.json') as f:
    tmp = json.load(f)
    x.extend(tmp)
    y.extend([1]*len(tmp))

with open(method+'/es_test_embedding.json') as f:
    tmp = json.load(f)
    x.extend(tmp)
    y.extend([2]*len(tmp))

with open(method+'/nl_test_embedding.json') as f:
    tmp = json.load(f)
    x.extend(tmp)
    y.extend([3]*len(tmp))

with open(method+'/ru_test_embedding.json') as f:
    tmp = json.load(f)
    x.extend(tmp)
    y.extend([4]*len(tmp))

pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
reduced_x = pca.fit_transform(x)  # 对样本进行降维
# reduced_x = np.dot(reduced_x, pca.components_) + pca.mean_  # 还原数据
print(silhouette_score(reduced_x,y))
print(calinski_harabasz_score(reduced_x, y))
# original
# -0.09008648209956029 264.0951029197697
# tl_embedding
# 1111 -0.062281309688654016 27.04297771520081
# 2222 -0.04700948892402741 21.652262041503526
# 3333 -0.06901430964203063 18.08615259011736
# 4444 -0.05175884902008767 18.530238225525622
# 5555 -0.052755365595452866 25.48921030858984
# avg 22.16
# sl_embedding
# 1111 -0.05682801819040762 30.0570681099818
# 2222 -0.057439485118375246 36.83737424894509
# 3333 -0.04682634279802796 16.401898430124653
# 4444 -0.04874186170590065 13.469265631528456
# 5555 -0.060296671531202925 18.68940998138701
# avg 23.09
# acsmtl_model_embedding
# 1111 -0.060083418337493595 18.165591864983334
# 2222 -0.07491861658650581 19.814337790548674
# 3333 -0.051763544445659636 38.24899599171688
# 4444 -0.06973153411300849 42.420604716352734
# 5555 -0.09361906055218096 68.36706906631142
# avg 37.40
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
yellow_x, yellow_y = [],[]
purple_x, purple_y = [],[]
# print(reduced_x)
for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    elif y[i] == 2:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
    elif y[i] == 3:
        yellow_x.append(reduced_x[i][0])
        yellow_y.append(reduced_x[i][1])
    else:
        purple_x.append(reduced_x[i][0])
        purple_y.append(reduced_x[i][1])


plt.figure(figsize=(8,8))
plt.scatter(red_x, red_y, c='coral', marker='.')
plt.scatter(blue_x, blue_y, c='cyan', marker='.')
plt.scatter(green_x, green_y, c='springgreen', marker='.')
plt.scatter(purple_x,purple_y,c='violet', marker='.')
plt.scatter(yellow_x,yellow_y,c='yellow',marker='.')
labelss = plt.legend(labels=["English","French","Spanish","Dutch","Russian"],loc="upper right",fontsize=10).get_texts()
[label.set_fontname('Times New Roman') for label in labelss]

for x,y,r in [(red_x,red_y,'r'),(blue_x,blue_y,'b'),(green_x,green_y,'g'),(yellow_x,yellow_y,'y'),(purple_x,purple_y,'m')]:
    red_x_center = sum(x) / len(x)
    red_y_center = sum(y) / len(y)
    red_distances = []
    for num,x_ in enumerate(x):
        y_ = y[num]
        red_distances.append(distance(red_x_center,x_,red_y_center,y_))
    max_red_distance = max(red_distances)
    # 8 8 30000
    # 8 8 36000 [-3, 3, -2, 3]
    # 8 8 5000 [-10, 15, -5, 6]
    plt.scatter(red_x_center, red_y_center,max_red_distance*8000,c=r,alpha=0.1)

plt.axis([-8, 10, -8, 10])
plt.show()