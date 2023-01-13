# -*- coding: utf-8 -*-
"""
GAN生成模型：GAN参数
"""

#%% 参数
# 路径参数
path_data = './SSEC_daily.xlsx' # 原始数据路径
path_suffix = 'ssec_daily_1y' # 后缀名路径
path_models = '../Models/' # 储存模型路径
path_results = '../Results/' # 储存结果路径

# 日期参数
date_train_start = '2004/12/31' # 训练集起始日期
date_train_end = '2020/03/31' # 训练集结束日期

# 数据预处理参数
use_col = 0 # 数据在表格的第几列（除日期列以外，从0开始）
use_ret = True # 使用收益（True）还是价格（False）
scale_ret = 100 # 收益放大多少倍：gan-100

# 网络训练参数
gan_type = 'gan' # gan
device = 'cuda' # cuda或cpu
num_epochs = 600
batch_size = 24 # Takahashi(2019)为24，Koshiyama(2019)为252
seed = 42 # 42
batch_norm = False # gan-False更好

# 网络模拟参数
num_gen = 100 # 生成假样本数

# 网络结构参数
if gan_type in ['gan']:
    # GAN网络结构参数，参考Takahashi(2019)
    gan_dim_latent = 100
    gan_dim_hidden_1 = 128
    gan_dim_hidden_2 = 128 # 2048
    gan_dim_output = 252 # 2520
    
