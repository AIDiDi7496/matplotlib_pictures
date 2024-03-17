#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')


# In[6]:


import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [40, 100, 30, 55]
bar_labels = ['red', 'blue', '_red', 'orange']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('fruit supply')
ax.set_title('Fruit supply by kind and color')
ax.legend(title='Fruit color')

plt.show()


# In[7]:


fig,ax= plt.subplots()


# In[17]:


fruits = ['apple','cherry','orange','blueberry']
counts=[40,100,30,55]
bar_label=['red','biue','_red','orange']
bar_color = ['tab:red','tab:blue','tab:red','tab:orange']
ax.bar(fruits,counts,label=bar_labels,color=bar_colors)


# In[18]:


plt.show()


# In[19]:


ax.set_ylabel('fruit supply')#设置y轴坐标
plt.show()


# In[20]:


ax.set_title('Fruit supply by kind and color')#为轴设置标题


# In[21]:


ax.legend(title='Fruit color')


# In[22]:


plt.show()


# In[25]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.size'] = 8.0  #设置全局的字体大小

# Fixing random state for reproducibility
np.random.seed(19680801)


# create random data
data1 = np.random.random([6, 50])

# set different colors for each set of positions
colors1 = ['C{}'.format(i) for i in range(6)]

# set different line properties for each set of positions
# note that some overlap
lineoffsets1 = [-15, -3, 1, 1.5, 6, 10]
linelengths1 = [5, 2, 1, 1, 3, 1.5]

fig, axs = plt.subplots(2, 2)

# create a horizontal plot
axs[0, 0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                    linelengths=linelengths1)

# create a vertical plot
axs[1, 0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                    linelengths=linelengths1, orientation='vertical')

# create another set of random data.
# the gamma distribution is only used for aesthetic purposes
data2 = np.random.gamma(4, size=[60, 50])

# use individual values for the parameters this time
# these values will be used for all data sets (except lineoffsets2, which
# sets the increment between each data set in this usage)
colors2 = 'black'
lineoffsets2 = 1
linelengths2 = 1

# create a horizontal plot
axs[0, 1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
                    linelengths=linelengths2)


# create a vertical plot
axs[1, 1].eventplot(data2, colors=colors2, lineoffsets=lineoffsets2,
                    linelengths=linelengths2, orientation='vertical')

plt.show()


# In[26]:


fig, axs = plt.subplots(2, 2)


# In[27]:


fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


# In[28]:


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


# In[29]:


fig, (ax1, ax2) = plt.subplots(1, 2)


# In[30]:


# First create some toy data:
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)

# Create just a figure and only one subplot
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')

# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)

# Create four polar axes and access them through the returned array
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar"))
axs[0, 0].plot(x, y)
axs[1, 1].scatter(x, y)

# Share a X axis with each column of subplots
plt.subplots(2, 2, sharex='col')

# Share a Y axis with each row of subplots
plt.subplots(2, 2, sharey='row')

# Share both X and Y axes with all subplots
plt.subplots(2, 2, sharex='all', sharey='all')

# Note that this is the same as
plt.subplots(2, 2, sharex=True, sharey=True)

# Create figure number 10 with a single subplot
# and clears it if it already exists.
fig, ax = plt.subplots(num=10, clear=True)


# In[31]:


fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Simple plot')
plt.show()


# In[33]:


# Create two subplots and unpack the output array immediately
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
ax1.plot(x, y)
ax1.set_title('Sharing Y axis')
ax2.scatter(x, y)
plt.show()


# In[35]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['font.size'] = 8.0  #设置全局的字体大小

# Fixing random state for reproducibility
np.random.seed(19680801)


# create random data
data1 = np.random.random([6, 50])

# set different colors for each set of positions
colors1 = ['C{}'.format(i) for i in range(6)]

# set different line properties for each set of positions
# note that some overlap
lineoffsets1 = [-15, -3, 1, 1.5, 6, 10]  #线条的偏移量
linelengths1 = [5, 2, 1, 1, 3, 1.5]      #线条的长度，
#第一个线条在-15的位置，长度为5

fig, axs = plt.subplots(2, 2)

# create a horizontal plot
axs[0, 0].eventplot(data1, colors=colors1, lineoffsets=lineoffsets1,
                    linelengths=linelengths1)
plt.show()


# In[47]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('C:/Users/86188/Desktop/stinkbug.png')#注意图像的地址
print(img)#图像数据通常以多维数组的形式表示，其中每个元素代表图像的一个像素点，包含了像素的颜色信息。


# In[48]:


imgplot = plt.imshow(img)


# In[55]:





# In[51]:


fig, ax = plt.subplots(figsize=(5, 2.7)) #figsize是图片的大小尺寸
ax.plot(data1, 'o', label='data1')
ax.plot(data2, 'd', label='data2')
ax.plot(data3, 'v', label='data3')
ax.plot(data4, 's', label='data4')
ax.legend();


# In[53]:


ax.set_title(r'$\sigma_i=15$')
#在 Matplotlib 中，如果要在标题中包含 LaTeX 公式，可以使用 $ 符号将公式括起来。


# In[61]:


#绘制折线图
import matplotlib.pyplot as plt
plt.plot([0,2],[1,4])#[0,2]是x轴上的两个点的位置，表示起点和终点
plt.show()


# In[72]:





# In[76]:


import matplotlib.pyplot as plt
import matplotlib

x = [1,2,3,4,5]
squares=[1,16,9,25,4]
#设置线条宽度
plt.plot(x,squares,linewidth=3)
#解决中文乱码问题
plt.rcParams["font.sans-serif"]=["SimHei"]
#添加标题
plt.title("标题设置",fontsize = 24)
#给x轴添加名称 
plt.xlabel("x轴",fontsize=18)
plt.ylabel("y轴",fontsize=18)
plt.show()


# In[79]:


import numpy as np
print(plt.style.available)#查看matplotlib有哪些风格


# In[98]:


import numpy as np 
import matplotlib.pyplot as plt
print(plt.style.available)
plt.style.use('Solarize_Light2')
plt.show()


# In[111]:


#自己画一个五天的气温图，记录最高和最低气温
import matplotlib.pyplot as plt
#构造数据
max_temp=[33,31,28,19,27]
min_temp=[10,19,17,9,11]
#生成x轴上面的5个刻度点
x=range(5) #注意返回的是0，1，2，3，4五个数字的整数序列，对应生成5个刻度点
#用列表推导式
x_ticks=[f'星期{i}'for i in range(1,6) ]
#构造标题
plt.title=("五天内的气温统计")
#给x轴添加名称
plt.xlabel("周")
#给y轴添加名称
plt.ylabel("温度，单位（°）")
plt.xticks(x,x_ticks)
#填充数据生成对应的日期
plt.plot(x,max_temp,label="最高温度")
plt.plot(x,min_temp,label="最低温度")
#显示图例
plt.legend(loc="upper left")
#这行代码表示的设置图例并且移动它的位置。
plt.style.use('seaborn-v0_8-whitegrid')
"""图例（legend）是用于解释图表中各个元素对应含义的标记。
Matplotlib 提供了多个位置选项来控制图例的位置，例如：
- `"upper right"`：右上角

- `"upper left"`：左上角

- `"lower right"`：右下角

- `"lower left"`：左下角

- `"center"`：居中
"""


# In[100]:


import matplotlib.pyplot as plt
import numpy as np
#画散点图
# 指定字体，例如微软雅黑
plt.rcParams['font.family'] = 'Microsoft YaHei'
#注意是由于画图时出现了负号，所以需要指定字体，防止出现乱码
x=np.linspace(0,10,100)#生成0-10之间的100个等差数，这个数组将被赋给变量x
plt.scatter(x,np.sin(x))
plt.show()


# In[ ]:


#复习：linspace(起始值，结束值，元素个数)


# In[118]:


import matplotlib.pyplot as plt

import numpy as np

#画不同大小颜色的散点图

np.random.seed(100)#设置了随机数生成器的种子

x=np.random.rand(100)

y=np.random.rand(100)

colors=np.random.rand(100)

size=np.random.rand(100)*1000  #用于表示散点图中每个点的大小

plt.scatter(x,y,c=colors,s=size,alpha=0.7)#alpha参数指定了点的透明度

plt.show()


# In[117]:


import matplotlib.pyplot as plt

import numpy as np

#将画布分为区域，将图画到画布的指定区域

x=np.linspace(1,10,100)

#将画布分为2行2列，将图画到画布的1区域

plt.subplot(2,2,1)

plt.plot(x,np.sin(x))

plt.subplot(2,2,3)

plt.plot(x,np.cos(x))

plt.show()


# In[109]:


import matplotlib.pyplot as plt
import numpy as np

# 将画布分为区域，将图画到画布的指定区域
x = np.linspace(1, 10, 100)

# 将画布分为2行2列，创建子图对象
fig, ax = plt.subplots(nrows=2, ncols=2)

# 在第一个子图(ax[0][1])中绘制曲线
ax[0][1].plot(x, np.sin(x))

# 在第二个子图(ax[1][1])中绘制曲线
ax[1][1].plot(x, np.cos(x))

# 显示图形
plt.show()


# In[90]:


import matplotlib.pyplot as plt
import numpy as np

# 设置同心圆的半径
radii = [1, 2, 3, 4, 5]

# 创建一个新的图形
fig, ax = plt.subplots()

# 循环绘制每个同心圆
for radius in radii:
    # 计算圆心坐标
    x = np.linspace(-radius, radius, 1000)
    y_positive = np.sqrt(radius**2 - x**2)
    y_negative = -np.sqrt(radius**2 - x**2)
    
    # 绘制散点
    ax.scatter(x, y_positive, color=(0.65, 0.25, 0.1), s=1)
    ax.scatter(x, y_negative, color=(0.65, 0.25, 0.1), s=1)

# 设置图形范围和纵横比
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')

# 移除坐标轴
ax.axis('off')

# 显示图形
plt.show()



# In[89]:


import matplotlib.pyplot as plt
import numpy as np

# 设置画布大小
plt.figure(figsize=(8, 8))

# 定义颜色
colors = [(1, 0.65, 0.25), (0.25, 0.25, 0.65)]  # (red, green, blue)，其中红色占10%，蓝色占65%

# 画15个同心圆环
for i in range(15):
    # 计算半径
    radius = i + 1
    # 绘制同心圆环
    circle = plt.Circle((0, 0), radius, color=colors[i % 2], alpha=0.6)
    plt.gca().add_patch(circle)

# 设置坐标轴范围和显示
plt.axis('scaled')
plt.axis('off')
plt.xlim(-16, 16)
plt.ylim(-16, 16)

# 显示图形
plt.show()


# In[96]:


import matplotlib.pyplot as plt
import numpy as np

# 设置同心圆的半径
radii = [0.5, 1.5, 2.5, 3.5, 4.5]

# 创建一个新的图形
fig, ax = plt.subplots()

# 循环绘制每个同心圆
for radius in radii:
    # 计算圆心坐标
    x = np.linspace(-radius, radius, 1000)
    y_positive = np.sqrt(radius**2 - x**2)
    y_negative = -np.sqrt(radius**2 - x**2)
    
    # 绘制散点
    ax.scatter(x, y_positive, color=(0.65, 0.25, 0.1), s=1)
    ax.scatter(x, y_negative, color=(0.65, 0.25, 0.1), s=1)

# 设置图形范围和纵横比
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')

# 移除坐标轴
ax.axis('off')

# 显示图形
plt.show()


# In[ ]:




