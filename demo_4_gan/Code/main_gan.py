# -*- coding: utf-8 -*-
"""
GAN和cGAN生成模型：主函数

"""

from Gan_Simulator import *

# 读取参数
import param_gan as param

# 创建路径
make_folders(param)

# 读取原始数据
raw_data = read_raw_data(param)

# 提取训练数据
train_data = extract_train_data(param,raw_data)

if param.gan_type=='gan':
    import core_gan
    # 构建并训练GAN
    G, res_train = core_gan.train_gan(param,train_data)
    
    # 使用GAN生成数据
    fake_data_gan = core_gan.simu_gan(param,G,train_data=train_data)

# 结果输出至excel
write_excel(param,res_train,train_data,fake_data_gan)

