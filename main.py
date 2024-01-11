import argparse
import os
import numpy as np
import pandas as pd
import math
from model.Generator_1 import Generator1
from model.Generator_2 import Generator2
from model.Discriminator_1 import Discriminator1
from model.Discriminator_2 import Discriminator2
import TranGAN1.tr_utils as tr_utils
import torch.optim as optim


import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--is_train", type=int, default=0, help="0:train 1:gennerate")
parser.add_argument("--dataset", type=str, default='phythonet', help="name of dataset")
parser.add_argument("--n_epochs", type=int, default=60, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--mask_rate", type=float, default=0.4, help="rate of mask number when testing")
parser.add_argument("--g1lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--g2lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--d1lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--d2lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--model_type", type=str, default='mean', help="number of model")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--log", type=int, default=1, help="serial number of model")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

if __name__ == '__main__':
    if opt.is_train == 0:
        print("is_train=={}".format(opt.is_train))
        """values = torch.Tensor([[1,2,3],[3,1,5]])
        values[0:1 , :] = 1
        print("values=={}".format(values))"""
        #tr_utils.get_condition(values)
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        data_path = './data'
        if opt.dataset == '301':
           data_obj, data_min, data_max = tr_utils.get_data(f'{data_path}/301/1-301规范格式过滤异常数据.csv', opt, device)
           dim = 7
           dim_tcn = 8
           n_class = 7
        elif opt.dataset == 'mimic':
           data_obj, data_min, data_max = tr_utils.get_data(f'{data_path}/mimic/vital_2.csv', opt, device)
           dim = 6
           dim_tcn = 7
           n_class = 6
        elif opt.dataset == 'phythonet':
           data_obj, data_min, data_max = tr_utils.get_data(f'{data_path}/phythonet/healthcare.csv', opt, device)
           dim = 6
           dim_tcn = 7
           n_class = 6
        train_loader = data_obj["train_dataloader_list"][0]
        test_loader = data_obj["test_dataloader_list"][0]
        len_test = len(train_loader)
        print("len_train=={}".format(len_test))
        hidden_dim = 128
        model_g1 = Generator2(dim, hidden_dim, n_class, n_layer=6).to(device)
        model_g2 = Generator1(dim_tcn, hidden_dim, n_class, n_layer=5).to(device)
        print(model_g2)
        model_d1 = Discriminator2(dim, hidden_dim, n_class, n_layer=6).to(device)
        model_d2 = Discriminator1(dim_tcn, hidden_dim, n_class, n_layer=5).to(device)
        optimizer_g1 = optim.Adam(model_g1.parameters(), lr=opt.g1lr)
        optimizer_g2 = optim.Adam(model_g2.parameters(), lr=opt.g2lr)
        torch.autograd.set_detect_anomaly(True)
        optimizer_d1 = optim.Adam(model_d1.parameters(), lr=opt.d1lr)
        optimizer_d2 = optim.Adam(model_d2.parameters(), lr=opt.d2lr)
        for epoch in range(opt.n_epochs):
            print("epoch:{}".format(epoch))
            print("进度:{:.10f}".format(epoch/opt.n_epochs))
            for ix, (train_batch, mask, len, raw, time, cond) in enumerate(train_loader):
                print("train=={}".format(train_batch))
                print("train=={}".format(train_batch.shape))
                # train generator
                #归一化cond条件
                cond_norm = tr_utils.normalize_cond(cond, data_min, data_max)
                train_batch = tr_utils.add_cond(train_batch , len , cond_norm)
                y_g1 = model_g1(train_batch)         
                y2 = y_g1.detach().clone()
                #根据掩码，用生成值填充缺失值，并保留原始值
                """print("y2=={}".format(y2))
                print("raw=={}".format(raw))
                combined_time1 = raw[1].cpu().numpy()
                np.savetxt('./zeros_array.csv', combined_time1, delimiter=',')"""
                y_2 = tr_utils.replace_raw(y2, mask, raw, len)
                y_2 = torch.cat((y_2, time.unsqueeze(-1)), 2)
                """print("y_2222222222222=={}".format(y_2-raw))
                combined_time2 = y_2[1].cpu().numpy()
                np.savetxt('./zeros.csv', combined_time2, delimiter=',')"""
                y_g2 = model_g2(y_2)
                # train discriminator
                y_3 = y_g2.detach().clone()
                #根据掩码，用生成值填充缺失值，并保留原始值
                y = tr_utils.replace_raw(y_3, mask, raw, len)
                y_33 = y.detach().clone()
                y_33 = tr_utils.add_cond(y_33 , len , cond)
                y_d1 = model_d1(y_33)
                y = torch.cat((y, time.unsqueeze(-1)), 2)
                y_d2 = model_d2(y)
                y_d11= y_d1.detach().clone()
                y_d22= y_d2.detach().clone()
                #print("y_d1=={}".format(y_d1))
                d1_loss = tr_utils.masked_bce_loss(y_d1, mask) + tr_utils.masked_bce_loss(y_d22, mask)
                d2_loss = tr_utils.masked_bce_loss(y_d2, mask) + tr_utils.masked_bce_loss(y_d11, mask)
                g1_loss = d1_loss * 10 + tr_utils.masked_mse_loss(raw, y_g1, mask, device='cuda') 
                g2_loss = d2_loss * 10 + tr_utils.masked_mse_loss(raw, y_g2, mask, device='cuda')
                """print("g1_loss=={:.10f}".format(g1_loss))
                print("g2_loss=={:.10f}".format(g2_loss))
                print("d1_loss=={:.10f}".format(d1_loss))"""
                #d1_loss = torch.nn.MSELoss(y_d1, mask)
                #print("y_g1=={}".format(y_g1))
                """print("g1_loss=={:.10f}".format(g1_loss))
                print("g2_loss=={:.10f}".format(g2_loss))
                print("d1_loss=={:.10f}".format(d1_loss))
                print("d2_loss=={:.10f}".format(d2_loss))"""
                optimizer_g1.zero_grad()
                g1_loss.backward(retain_graph=True)
                optimizer_g1.step()
                optimizer_g2.zero_grad()   
                #print("y_g1=={}".format(y_g2))
                #print("g2_loss=={:.10f}".format(g2_loss))
                g2_loss.backward(retain_graph=True)
                optimizer_g2.step()
                optimizer_d1.zero_grad()
                d1_loss.backward(retain_graph=True)
                optimizer_d1.step()
                optimizer_d2.zero_grad()
                d2_loss.backward(retain_graph=True)
                optimizer_d2.step()
                
                
                #print("ix=={}".format(ix))
                #if ix == 2:
                #print("label=={}".format(label))
                #print("train_batch={}".format(train_batch[0]))
                #print("mask=={}".format(mask))
                #print("len=={}".format(len))
                #train generater        
                
                """optimizer_g1.zero_grad()
                y_g1 = model_g1(train_batch)
                g1_loss = tr_utils.masked_mse_loss(raw, y_g1, mask,device='cuda')
                g1_loss.backward(retain_graph=True)
                optimizer_g1.step()
                
                
                #train discrimnater
                optimizer_d1.zero_grad()
                y = train_batch
                y_d1 = model_d1(y)
                d1_loss = tr_utils.masked_bce_loss(y_d1, mask)
                #d1_loss = tr_utils.masked_mse_loss(raw, y_g1, mask,device='cuda')
                print("d1_loss=={}".format(d1_loss))
                d1_loss.backward(retain_graph=True)
                optimizer_d1.step()"""
                #print("y_pred=={}".format(y_pred.shape))#(64,4),但我需要与len长度一样，特征数也一样的数据
                #x_1 = tr_utils.intercept(y_pred,len)
                #print("x_1=={}".format(x_1.shape))         
                """test = np.array(train_batch[0])
                df = pd.DataFrame(test)
                df.to_csv("./test.csv")"""
                #print(train_batch.shape)#(64,202,7)
            if epoch >= 0:
                g1_state_dict = model_g1.state_dict()
                g2_state_dict = model_g2.state_dict()
                d1_state_dict = model_d1.state_dict()      
                torch.save({
                        'args': opt,
                        'g1_state_dict': g1_state_dict,
                        'g2_state_dict': g2_state_dict,
                        'd1_state_dict': d1_state_dict,
                        }, f'./save_model/model_{opt.dataset}_{epoch}.model')
    if opt.is_train == 1:
        #modelg1, modelg2, modeld1 = tr_utils.load_model(f'./savemodel/model_{opt.dataset}_82.model', opt)
        modelg1, modelg2, modeld1 = tr_utils.load_model(f'./best_model/model_mimic_mimic_17.model', opt)
        modelg2.eval()
        modelg1.eval()
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        data_path = './data'
        attr_list = ['心率', '呼吸率', '收缩压', '舒张压', '氧饱和度', '格拉斯评分', '体温']
        if opt.dataset == '301':
            data_obj, data_min, data_max = tr_utils.get_data(f'{data_path}/301/1-301规范格式过滤异常数据.csv', opt, device)
        elif opt.dataset == 'mimic':
            print("|||||||||||||||||||||||||||||||||||")
            data_obj, data_min, data_max = tr_utils.get_data(f'{data_path}/mimic/vital_2.csv', opt, device)
        elif opt.dataset == 'phythonet':
            data_obj, data_min, data_max = tr_utils.get_data(f'{data_path}/phythonet/healthcare.csv', opt, device)
        
        train_loader = data_obj["train_dataloader_list"][0]
        test_loader = data_obj["test_dataloader_list"][0]
        len_test = len(test_loader)
        """print("len_test=={}".format(len_test))
        print("n_train=={}".format(data_obj["n_train_batch"]))
        print("n_test=={}".format(data_obj["n_test_batch"]))"""
        mae_m = 0.0
        mae_x = 0
        rmse_m = 0.0
        rmse_count = 0
        mre_count = 0.0
        mre_m = 0.0
        if opt.model_type == 'T2GAN':
            
            for ix, (train_batch, mask, len, raw, time, cond) in enumerate(test_loader):         
                #if ix == 1: break
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                for iy in range(6):
                    if iy == 0:
                        #生成用于标记未填充部分的mask
                        cond_norm = tr_utils.normalize_cond(cond, data_min, data_max)
                        train_batch = tr_utils.add_cond(train_batch , len , cond_norm)
                        test_mask1,test_mask2 = tr_utils.set_testmask(mask,opt.mask_rate)
                        noimpute_mask = tr_utils.generate_noimpute(test_mask1, len)
                        mask1 = test_mask1.clone()
                        for dx in range(len.shape[0]):
                            mask1[dx, int(len[dx,0]):int(len[dx,0])+2, :] = 1
                        train_batch = train_batch * mask1
                        raw_initial = raw * test_mask1 #后面raw用于计算MAE，这里的raw_initial作为已知的初始原始值
                        y_g2 = modelg1(train_batch)
                        y3 = y_g2.detach().clone()
                        y4 = y3.detach().clone()
                        y_2 = tr_utils.replace_raw(y3,test_mask1, raw_initial, len)
                        y_2 = torch.cat((y_2, time.unsqueeze(-1)), 2)
                        y_g3 = modelg2(y_2)
                        y5 = y_g3.detach().clone()
                        col_impute = tr_utils.multistage(noimpute_mask,5-iy)
                        raw_impute = y5 * col_impute
                        norm_impute =((raw_impute- data_min.to(device)) / data_max.to(device))*col_impute
                        """mae = tr_utils.caculate_mae(y5, test_mask2, len, raw)
                        #print("rawloop66666=={}".format(raw_loop))
                        #print("y5=={}".format(y5))
                        #print("raw_impute=={}".format(raw_impute))
                        #print("test_mask2=={}".format(test_mask2))
                        print("mae=={}".format(mae))
                        mae_m = mae_m + mae
                        mae_x = mae_x + 1"""
                        """print("raw_impute=={}".format(raw_impute))
                        print("norm_impute=={}".format(norm_impute))
                        print("train=={}".format(train_batch))"""
                        #更新迭代数据
                        train_batch = train_batch + norm_impute
                        raw_loop = raw_initial + raw_impute
                        test_mask1 = tr_utils.let_i(test_mask1, len, 5-iy)#令test_mask1中第iy列数据为1

                        """test_mask22 = tr_utils.multistage(test_mask2,6-iy)
                        print("testmask2222=={}".format(test_mask22.sum()))
                        mae = tr_utils.caculate_mae(raw_impute, test_mask22, len, raw)
                        print("mae00000=={}".format(mae))"""
                        #train_batch[train_batch>0] = 1
                        """print("train=={}".format(train_batch)) 
                        print("test_mask1=={}".format(test_mask1 - train_batch))  """    
                    elif iy == 5:
                        #生成用于标记未填充部分的mask
                        noimpute_mask = tr_utils.generate_noimpute(test_mask1, len)
                        train_batch = tr_utils.add_cond(train_batch , len , cond_norm)
                        #print("train_batch=={}".format(train_batch))
                        y_g2 = modelg1(train_batch)
                        y3 = y_g2.detach().clone()
                        y4 = y3.detach().clone()
                        y_2 = tr_utils.replace_raw(y3,test_mask1, raw_loop, len)
                        y_2 = torch.cat((y_2, time.unsqueeze(-1)), 2)
                        y_g3 = modelg2(y_2)
                        y5 = y_g3.detach().clone()

                        col_impute = tr_utils.multistage(noimpute_mask,5-iy)
                        raw_impute = y5 * col_impute
                        #print(iy)
                        #print("raw_impute=={}".format(raw_impute))
                        raw_loop = raw_loop + raw_impute
                        #print("rawloop=={}".format(raw_loop))
                        #mae = tr_utils.caculate_mae(raw_loop, test_mask2, len, raw)
                        mae, count = tr_utils.caculate_mae(raw_loop, test_mask2, len, raw, data_max, data_min,device=device)
                        #print("rawloop66666=={}".format(raw_loop))
                        #print("y5=={}".format(y5))
                        #print("raw_impute=={}".format(raw_impute))
                        #print("test_mask2=={}".format(test_mask2))
                        print("mae=={}".format(mae))
                        mae_m = mae_m + mae
                        mae_x = mae_x + count
                    else:
                        #print("raw=={}".format(raw))
                        #生成用于标记未填充部分的mask
                        noimpute_mask = tr_utils.generate_noimpute(test_mask1, len)
                        train_batch = tr_utils.add_cond(train_batch , len , cond_norm)
                        y_g2 = modelg1(train_batch)
                        y3 = y_g2.detach().clone()
                        y4 = y3.detach().clone()
                        y_2 = tr_utils.replace_raw(y3,test_mask1, raw_loop, len)
                        y_2 = torch.cat((y_2, time.unsqueeze(-1)), 2)
                        y_g3 = modelg2(y_2)
                        y5 = y_g3.detach().clone()
                        col_impute = tr_utils.multistage(noimpute_mask,5-iy)
                        raw_impute = y5 * col_impute
                        norm_impute =((raw_impute- data_min.to(device)) / data_max.to(device))*col_impute
                        #更新迭代数据
                        train_batch = train_batch + norm_impute
                        raw_loop = raw_loop + raw_impute             
                        #print("rawloop=={}".format(raw_loop))
                        test_mask1 = tr_utils.let_i(test_mask1, len, 5-iy)#令test_mask1中第iy列数据为1
        
            mae_r = mae_m / mae_x
            print(mae_m)
            print(mae_x)
            print(mae_r)
        elif opt.model_type == 'mean':
            for ix, (train_batch, mask, len, raw, time, cond) in enumerate(test_loader): 
                #print(f"train_batch======={train_batch.shape}")        
                #if ix == 1: break
                train_batch = train_batch * data_max.to(device) + data_min.to(device)
                #print(f"train_batch======={train_batch}")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                test_mask1,test_mask2 = tr_utils.set_testmask(mask,opt.mask_rate)
                train_raw = train_batch * test_mask1   #得到人工掩码后的数据
                mean_batch = tr_utils.mean_impute(train_raw, test_mask1, test_mask2, data_min, data_max)
                #print(f"mean_batch======{mean_batch}")
                mean_batch = (mean_batch - data_min.to(device)) / ((data_max.to(device)) -data_min.to(device))
                norm_raw = train_batch * test_mask2
                #print(f"norm_raw======{norm_raw}")

                norm_raw = (norm_raw- data_min.to(device)) / ((data_max.to(device)) -data_min.to(device))
                differ_mae =  (mean_batch - norm_raw) * test_mask2
                #print(f"differ_mae==={differ_mae}")
                impute_sum = test_mask2.sum()
                
                mae = torch.abs(differ_mae).sum() / impute_sum
                mre_count = mean_batch
                rmse_count = rmse_count + impute_sum
                rmse = torch.square(differ_mae).sum()
                rmse_m = rmse_m + rmse
                #print(f"mae==={mae}")
                mae_m = mae_m + mae
                mae_x = mae_x + 1
            mae_r = mae_m / mae_x
            rmse_r = rmse_m / rmse_count
            print(mae_m)
            print(mae_x)
            print(f"mae:{mae_r}")
            print(f"rmse_r:{rmse_r.sqrt()}")
        
        