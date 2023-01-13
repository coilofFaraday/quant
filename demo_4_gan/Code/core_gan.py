# -*- coding: utf-8 -*-
"""
GAN生成模型：gan

参考资料：
    参考Takahashi et al. 2019. Modeling financial time-series with generative adversarial networks
    https://github.com/stakahashy/fingan/

函数说明：
    build_gan_generator 构建GAN生成器
    build_gan_discriminator 构建GAN判别器
    train_gan 训练GAN
    simu_gan GAN生成序列   
    
"""

import numpy as np

import torch
import torch.nn as nn

from Gan_Simulator import set_random_seed, get_loader, plot_save_train, plot_simu


#%% 构建GAN生成器
class build_gan_generator(nn.Module):
    """
    GAN生成器结构：
    输入层：gan_dim_latent个随机数（batch_size*gan_doim_latent）
    第i隐藏层：gan_dim_hidden_i个Tanh神经元
    输出层：gan_dim_output
    """
    def __init__(self,param): # 类初始化
        super(build_gan_generator,self).__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.Tanh())
            return layers
        
        self.Net = nn.Sequential(
            *block(param.gan_dim_latent, param.gan_dim_hidden_1),
            *block(param.gan_dim_hidden_1, param.gan_dim_hidden_2),
            nn.Linear(param.gan_dim_hidden_2, param.gan_dim_output),
            )

    def forward(self,x): # 前向传播
        out = self.Net(x)
    
        return out      


#%% 构建GAN判别器，均取原文参数
class build_gan_discriminator(nn.Module):
    """
    GAN判别器结构：
    输入层：gan_dim_output个真实值/生成器生成值
    第i隐藏层：gan_dim_hidden_i个Tanh神经元
    输出层：1个Sigmoid神经元
    """
    def __init__(self,param): # 类初始化
        super(build_gan_discriminator,self).__init__()
        
        kernel_size = 9
        
        def block(in_chan, out_chan, normalize=True):
            layers = [nn.Conv1d(in_chan, out_chan, kernel_size=kernel_size, padding=(kernel_size-1)//2)]
            if param.batch_norm:
                layers.append(nn.BatchNorm1d(out_chan))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers
        
        self.Net = nn.Sequential(
            *block(1, 64, normalize=False),
            *block(64, 128, normalize=False),
            *block(128, 128, normalize=False),
            nn.Flatten(),
            nn.Linear(128*param.gan_dim_output, 32),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(32,1),
            "nn.Sigmoid()"
            )
            
    def forward(self,x): # 前向传播
        out = self.Net(x)
        return out  


#%% 训练GAN
def train_gan(param,train_data):
    """
    训练GAN，返回生成器G和训练损失结果res

    Parameters
    ----------
    param : Class(Param)
        参数类.
    train_data : T*1 ndarray
        收益率或价格序列.

    Returns
    -------
    G : torch.nn.Module的子类
        生成器.
    res : num_epochs*2 DataFrame
        d_loss和g_loss.

    """
    # 0. 设置随机数种子点
    set_random_seed(param.seed)
    
    # 1. 定义数据生成器
    train_loader = get_loader(param,
                              train_data,
                              window_width=param.gan_dim_output,
                              batch_size=param.batch_size,
                              shuffle=True)
    
    # 2. 定义网络结构
    D = build_gan_discriminator(param).to(param.device)
    G = build_gan_generator(param).to(param.device)
    
    # 3. 定义损失函数和优化器
    """def compute_gradient_penalty_1(D, real_data, fake_data,lambda_term,batch_size,cuda_index):
        tensor=torch.FloatTensor(batch_size,1,1,1).uniform_(0,1)
        tensor=tensor.expand(batch_size,real_data.size(1),real_data.size(2),real_data.size(3))
        if cuda:
            tensor=tensor.cuda(cuda_index)
        else:
            tensor=tensor
        interpolated=tensor*real_data+((1-tensor)*fake_data)
        if cuda:
            interpolated= interpolated.cuda(cuda_index)
        else:
            interpolated=interpolated
        interpolated=Variable(interpolated,requires_grad=True)
        d_interpolated=D(interpolated)
        grads=autograd.grad(outputs=d_interpolated,inputs= interpolated,grad_outputs=torch.ones(d_interpolated.size()).cuda(cuda_index)if cuda else torch.ones(d_interpolated.size()),create_graph=True, retain_graph=True)[0]
        grad_penalty=((grads.norm(2,dim=1)-1)**2).mean()*lambda_term
        return grad_penalty"""
    def compute_gradient_penalty_2(D,real_data,fake_data,lambda_term,cuda_index):
        batch_size=real_data.size()[0]
        tensor=torch.rand(batch_size,1,1)
        tensor=tensor.expand_as(real_data)
        tensor=tensor.cuda(cuda_index)
        interpolated=tensor*real_data+((1-tensor)*fake_data)
        interpolated=interpolated.cuda(cuda_index)
        interpolated=torch.autograd.Variable(interpolated,requires_grad=True)
        d_interpolated=D(interpolated)
        grads=torch.autograd.grad(outputs=d_interpolated,inputs= interpolated,grad_outputs=torch.ones(d_interpolated.size()).cuda(cuda_index),create_graph=True, retain_graph=True)[0]
        grad_penalty=((grads.norm(2,dim=1)-1)**2).mean()*lambda_term
        return grad_penalty
    d_optimizer = torch.optim.Adam(D.parameters(), lr=1e-5, betas=(0.1,0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))

    # 4. 训练
    # 初始化结果
    res = {'d_loss':np.zeros((param.num_epochs)),
           'g_loss':np.zeros((param.num_epochs))}
    # 训练模式
    D.train()
    G.train()
    # 逐轮迭代
    for epoch in range(param.num_epochs):
        # a. 训练判别器
        # a1. 生成真样本
        real_data = next(iter(train_loader)).to(param.device)
        # 判别器含卷积操作，需要三维数据，故真样本二维转三维
        real_data = torch.reshape(real_data,(real_data.shape[0],1,real_data.shape[1]))
        # a2. 生成假样本
        z = torch.randn(param.batch_size, param.gan_dim_latent).to(param.device)
        fake_data = G(z)
        # 判别器含卷积操作，需要三维数据，故假样本二维转三维
        fake_data = torch.reshape(fake_data,(fake_data.shape[0],1,fake_data.shape[1]))
        # a3. 生成真假标签，noise labeling
        real_label = (torch.rand(param.batch_size)/5+0.9).to(param.device)
        fake_label = (torch.rand(param.batch_size)/5+0.1).to(param.device)
        real_label = real_label.reshape((-1,1))
        fake_label = fake_label.reshape((-1,1))
        # a4. 判别器进行预测
        real_pred = D(real_data)
        fake_pred = D(fake_data)
        # a5. 计算损失
        d_loss_fake = fake_pred.mean()
        d_loss_real = real_pred.mean()
        d_loss_grad = compute_gradient_penalty_2(D, real_data, fake_data,10,0)
        d_loss = d_loss_fake - d_loss_real + d_loss_grad
        # a6. 训练判别器
        d_optimizer.zero_grad() # 梯度清零
        d_loss.backward() # 反向传播计算梯度
        d_optimizer.step() # 更新参数
        
        # b. 训练生成器
        # b1. 生成假样本
        z = torch.randn(param.batch_size, param.gan_dim_latent).to(param.device)
        fake_data = G(z)
        # 判别器含卷积操作，需要三维数据，故假样本二维转三维
        fake_data = torch.reshape(fake_data,(fake_data.shape[0],1,fake_data.shape[1]))
        # b2. 判别器进行预测
        fake_pred = D(fake_data)
        # b3. 计算损失
        # 不做noise labeling
        real_label = torch.ones(param.batch_size,1).to(param.device)
        g_loss = -fake_pred.mean()
        # b4. 训练生成器
        g_optimizer.zero_grad() # 梯度清零
        g_loss.backward() # 反向传播计算梯度
        g_optimizer.step() # 更新参数

        # c. 记录结果并打印
        res['d_loss'][epoch] = d_loss.item()
        res['g_loss'][epoch] = g_loss.item()
        print('Epoch [{}/{}], d_loss: {:.6f}, g_loss: {:.6f} '.
              format(epoch+1, param.num_epochs, 
                     d_loss.item(), g_loss.item()))

    # 5. 绘图和保存
    plot_save_train(param,D,G,res)
    # 返回生成器和结果
    return G, res


#%% GAN生成序列
def simu_gan(param,G,train_data=None):
    """
    返回GAN生成数据

    Parameters
    ----------
    param : Class(Param)
        参数类.
    G : torch.nn.Module的子类
        生成器.
    train_data : T*1 ndarray, optional
        收益率或价格序列. The default is None.

    Returns
    -------
    fake_data : N*seq_lengths ndarray
        若param.use_ret为True，返回收益率.
        若param.use_ret为False，返回标准化价格.

    """
    # 0. 设置随机数种子点
    set_random_seed(param.seed)
    # 测试模式
    G.eval()
    
    # 1. 生成num_gen条假样本
    if param.gan_type == 'gan':
        seq_lengths = param.gan_dim_latent
    z = torch.randn(param.num_gen, seq_lengths).to(param.device)
    fake_data = G(z).detach().cpu().numpy()
    
    # 2. 数据处理
    if param.use_ret:
        # 若数据为收益率，缩小一定倍数
        fake_data = fake_data / param.scale_ret
    else:
        # 若数据为价格，直接返回标准化价格
        pass

    # 3. 绘制生成结果
    plot_simu(param,fake_data,train_data)
    
    # 返回生成数据
    return fake_data

