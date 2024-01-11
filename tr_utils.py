import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model.Generator_1 import Generator1
from model.Generator_2 import Generator2
from model.Discriminator_1 import Discriminator1
from model.Discriminator_2 import Discriminator2
from sklearn import model_selection, metrics
from sklearn.metrics import confusion_matrix
import json
#from focal_loss import focal_losss
#在数据后加入cond条件
def add_cond(data_batch , len , cond):
    len_batch = data_batch.shape[0] #64
    dim_matrix = data_batch.shape[2]
    for ix in range(len_batch):
        data_batch[ix, int(len[ix,0]):int(len[ix,0])+2, 0:dim_matrix]  = cond[ix, :, :]
    return data_batch

def d1_cond(y_batch, len, cond):
    len_batch = y_batch.shape[0] #64
    dim_matrix = y_batch.shape[2]
    for ix in range(len_batch):
        y_batch[ix, int(len[ix,0]):int(len[ix,0])+2, 0:dim_matrix]  = cond[ix, :, :]
    return y_batch
def normalize_cond(data, att_min, att_max):
    #print("data=={}".format(data))
    mask = (data!=0).int()
    att_max[att_max == 0.] = 1.
    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!") 
    if torch.isnan(data_norm).any():
        raise Exception("nans!")
    # set masked out elements back to zero
    data_norm[mask == 0] = 0
    #print("data_norm3333=={}".format(data_norm))
    return data_norm

#根据缺失部分掩码，生成单列的掩码，用于多阶段生成任务。
def multistage(orignal_mask, col):
     new_mask = torch.zeros_like(orignal_mask)
     new_mask[:, :, col] = orignal_mask[:, :, col] # 将指定列设置为原掩码中的相应值
     return new_mask
     


#根据缺失率计算新的掩码（用于测试集上的数据掩盖）
def set_testmask(mask,mask_rate):
    ones_positions = (mask == 1).nonzero(as_tuple=False)
    # 计算需要更改的1的数量
    num_ones_to_change = int(mask_rate * ones_positions.size(0))
    # 随机选择需要更改的1的位置
    indices_to_change = ones_positions[torch.randperm(ones_positions.size(0))[:num_ones_to_change]]
    # 创建新的掩码张量
    mask1 = mask.clone()
    mask2 = torch.zeros_like(mask)
    # 将选中的位置的值更改为0
    for idx in indices_to_change:
        mask1[idx[0], idx[1], idx[2]] = 0
        mask2[idx[0], idx[1], idx[2]] = 1
    """print("1111=={}".format(ones_positions))
    print("2222=={}".format(num_ones_to_change))
    print("33333=={}".format(indices_to_change))"""
    #mask1用于对train_batch的掩盖，mask2用于计算平均绝对误差
    return mask1, mask2
#生成对应的dataframe数据
def collect_data(train_batch, len):
    
    return 0
    

#平均值填充方法，返回填充的数据矩阵
def mean_impute(train_raw, mask1, mask2, data_min, data_max):
    len_batch = train_raw.shape[0] #64
    mask3 = mask2.clone().to('cuda')
    for dx in range(len_batch):
         sum_column_0 = torch.sum(train_raw[dx, :, 0])
         sum_column_1 = torch.sum(train_raw[dx, :, 1])
         sum_column_2 = torch.sum(train_raw[dx, :, 2])
         sum_column_3 = torch.sum(train_raw[dx, :, 3])
         sum_column_4 = torch.sum(train_raw[dx, :, 4])
         sum_column_5 = torch.sum(train_raw[dx, :, 5])

         sum_mask_0 = torch.sum(mask1[dx, :, 0])
         sum_mask_1 = torch.sum(mask1[dx, :, 1])
         sum_mask_2 = torch.sum(mask1[dx, :, 2])
         sum_mask_3 = torch.sum(mask1[dx, :, 3])
         sum_mask_4 = torch.sum(mask1[dx, :, 4])
         sum_mask_5 = torch.sum(mask1[dx, :, 5])
         if sum_mask_0 != 0:
            column_0 = sum_column_0 / sum_mask_0
         else:
            column_0 = (data_max[0] + data_min[0])/2
            #column_0 = 0
         if sum_mask_1 != 0:
            column_1 = sum_column_1 / sum_mask_1
         else:
            column_1 = (data_max[1] + data_min[1])/2
            #column_1 = 0
         if sum_mask_2 != 0:
            column_2 = sum_column_2 / sum_mask_2
         else:
            column_2 = (data_max[2] + data_min[2])/2
            #column_2 = 0
         if sum_mask_3 != 0:
            column_3 = sum_column_3 / sum_mask_3
         else:
            column_3 = (data_max[3] + data_min[3])/2
            #column_3 = 0
         if sum_mask_4 != 0:
            column_4 = sum_column_4 / sum_mask_4        
         else:
            column_4 = (data_max[4] + data_min[4])/2
            #column_4 = 0
         if sum_mask_5 != 0:
             column_5 = sum_column_5 / sum_mask_5
         else:
            column_5 = (data_max[5] + data_min[5])/2
            #column_5 = 0
    
         mask3[dx, mask3[dx, :, 0]==1,0]  = column_0.to('cuda')
         mask3[dx, mask3[dx, :, 1]==1,1]  = column_1.to('cuda')
         mask3[dx, mask3[dx, :, 2]==1,2]  = column_2.to('cuda')
         mask3[dx, mask3[dx, :, 3]==1,3]  = column_3.to('cuda')
         mask3[dx, mask3[dx, :, 4]==1,4]  = column_4.to('cuda')
         mask3[dx, mask3[dx, :, 5]==1,5]  = column_5.to('cuda')             
             
         #print(f"mean==={mask3}")
    return mask3
#Intercept data
def intercept(data , len):
    return data[:,0:len,:]

#生成用于标记未填充部分的mask,依据数据长度的逻辑相反矩阵
def generate_noimpute(mask1, len):
    len_batch = mask1.shape[0] #64
    dim_matrix = mask1.shape[2]
    mask = torch.zeros_like(mask1)
    for ix in range(len_batch):
        mask[ix, 0:int(len[ix,0]), 0:dim_matrix]  = 1 - mask1[ix, 0:int(len[ix,0]), 0:dim_matrix] 
    return mask

#将mask第i列赋值为1，依据原始数据的大小len
def let_i(test_mask, len, iy):
    len_batch = test_mask.shape[0] #64
    for ix in range(len_batch):
        test_mask[ix, 0:int(len[ix,0]), iy]  = 1
    return test_mask
     
#根据掩码，用生成值填充缺失值，并保留原始值
def replace_raw(y, mask, raw, len):
    len_batch = y.shape[0] #64
    dim_matrix = y.shape[2]
    for ix in range(len_batch):
        raw_a = raw[ix, 0:int(len[ix,0]), 0:dim_matrix]
        mask_1 = mask[ix, 0:int(len[ix,0]), 0:dim_matrix] 
        y_1 = y[ix, 0:int(len[ix,0]), 0:dim_matrix]
        """combined_time1 = y_1.mul( 1 - mask_1 ).cpu().numpy()
        combined_time2 = raw_a.mul(mask_1).cpu().numpy()
        print("1111111111111={}".format(y_1.mul( 1 - mask_1 )))
        print("2222222222222={}".format(raw_a.mul(mask_1)))
        np.savetxt('./z.csv', combined_time1, delimiter=',')
        np.savetxt('./zray.csv', combined_time2, delimiter=',')"""
        y_1 = raw_a.mul(mask_1) + y_1.mul( 1 - mask_1 )
        """combined_time3 = y_1.cpu().numpy()
        np.savetxt('./ray.csv', combined_time3, delimiter=',')"""
        y[ix, 0:int(len[ix,0]), 0:dim_matrix] = y_1
        """combined_time4 = y[ix, 0:int(len[ix,0]), 0:dim_matrix].cpu().numpy()
        np.savetxt('./ay.csv', combined_time4, delimiter=',')"""
    return y

#测试集部分，根据掩码，用生成值填充缺失值，并保留原始值
def replace_raw_test(y, mask, raw, len):
    len_batch = y.shape[0] #64
    dim_matrix = y.shape[2]
    for ix in range(len_batch):
        raw_a = raw[ix, 0:int(len[ix,0]), 0:dim_matrix]
        mask_1 = mask[ix, 0:int(len[ix,0]), 0:dim_matrix] 
        y_1 = y[ix, 0:int(len[ix,0]), 0:dim_matrix]
        y_1 = raw_a + y_1.mul( 1 - mask_1 )
        y[ix, 0:int(len[ix,0]), 0:dim_matrix] = y_1
    return y

#Calculate MAE for imputation data
# def caculate_mae(data_batch, mask, len, raw, data_max, data_min,device):
#     mean_batch = data_batch * mask
#     mean_batch = (mean_batch - data_min.to(device)) / ((data_max.to(device)) -data_min.to(device))

#     norm_raw = raw * mask
#     norm_raw = (norm_raw- data_min.to(device)) / ((data_max.to(device)) -data_min.to(device))
#     differ_mae =  (mean_batch - norm_raw) * mask
#     impute_sum = mask.sum()
#     mae = torch.abs(differ_mae).sum() / impute_sum
#     return mae
   
#Calculate MAE for imputation data
def caculate_mae(data_batch, mask, len, raw, data_max, data_min, device):
    len_batch = data_batch.shape[0] #64
    dim_matrix = data_batch.shape[2]
    mae = 0
    count = 0
    #print("len=={}".format(len[0,0]))
    for ix in range(len_batch):
        data = data_batch[ix, 0:int(len[ix,0]), 0:dim_matrix]
        raw_a = raw[ix, 0:int(len[ix,0]), 0:dim_matrix]
        mask_1 = mask[ix, 0:int(len[ix,0]), 0:dim_matrix] 
        mask_2 = mask_1.clone()
        raw_1 = raw_a * mask_2
        data_1 = data * mask_2
        impute_sum = mask_2.sum()
       
        if impute_sum == 0:
            mae_count = 0
        else:
            differ = ((data_1-raw_1) / ((data_max.to(device)) - data_min.to(device))) * mask_2
            mae_count = torch.abs(differ).sum() 
        mae = mae + mae_count
        count = count + impute_sum
        # if impute_sum == 0:
        #     mae_count = 0
        # else:
        #     differ_mae = ((data_1- raw_1) / ((data_max.to(device)) - data_min.to(device))) * mask_2
        #     #norm_raw = ((raw_1- data_min.to(device)) / ((data_max.to(device)) -data_min.to(device))) * mask_2
        #     #differ_mae = norm_data - norm_raw
        #     print(f"differ====={differ_mae}")
        #     print(f"====={(data_max.to(device)) - data_min.to(device)}")
        #     mae_count = torch.abs(differ_mae).sum()
        # mae = mae + mae_count
        
    return mae, count
    
#load model
def load_model(model_path, arg, device='cuda'):
    if device == 'cpu':
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(model_path)
    if arg.dataset == '301':
        # 生命体征个数
        dim = 7
        dim_tcn = 8
        n_class = 7
    elif arg.dataset == 'mimic':
        dim = 6
        dim_tcn = 7
        n_class = 6
    elif arg.dataset == 'phythonet':
        dim = 6
        dim_tcn = 7
        n_class = 6
    hidden_dim = 128
    model_g1 = Generator2(dim, hidden_dim, n_class, n_layer=6).to(device)
    model_g2 = Generator1(dim_tcn, hidden_dim, n_class, n_layer=5).to(device)
    model_d1 = Discriminator2(dim, hidden_dim, n_class, n_layer=6).to(device)

    model_g1.load_state_dict(checkpoint['g1_state_dict'])
    model_g2.load_state_dict(checkpoint['g2_state_dict'])
    model_d1.load_state_dict(checkpoint['d1_state_dict'])
    return model_g1, model_g2, model_d1
    
   

#MSE loss
def masked_mse_loss(original_mat, generated_mat, mask_mat ,device):
    # 应用掩码
    """original_mat = original_mat.detach().cpu().numpy()
    generated_mat = generated_mat.detach().cpu().numpy()
    mask_mat = mask_mat.detach().cpu().numpy()"""
    #print("mask=={}".format(mask_mat))
    original_mat = original_mat * mask_mat
    #print("gener222=={}".format(generated_mat))
    generated_mat = generated_mat * mask_mat
    #print("gener222=={}".format(generated_mat))
    # 计算MSE损失
    mse_loss = torch.mean((original_mat - generated_mat)**2)
    return mse_loss

#cross entropy
def masked_bce_loss(output_mat, mask_mat):
    # 应用 sigmoid 函数以确保输出在 [0, 1] 范围内
    output = output_mat
    # 计算掩码下的二元交叉熵损失
    bce_loss = F.binary_cross_entropy(output, mask_mat, reduction='none')
    
    # 应用掩码
    masked_bce_loss = bce_loss * mask_mat
    num_valid_samples = mask_mat.sum().float()  # 有效样本数量
    #print("num_valid_samples=={}".format(num_valid_samples))
    #print("mask_mat=={}".format(mask_mat[1]))
    #print("bce_loss=={}".format(bce_loss))
    masked_bce_loss = bce_loss * mask_mat.float()  # 将无效样本的损失置零
    mean_bce_loss = torch.sum(masked_bce_loss)

    # 计算平均损失
    return mean_bce_loss

def get_data_min_max(records, device):
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	data_min, data_max = None, None
	inf = torch.Tensor([float("Inf")])[0].to(device)

	for b, (id, tt, vals, mask, labels) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max

def get_mimic_min_max(records, device):
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	data_min, data_max = None, None
	inf = torch.Tensor([float("Inf")])[0].to(device)

	for b, (time, vals, mask) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max


def normalize_masked_data_2(data, mask, att_min, att_max):
    # we don't want to divide by zero
    #print("att_max2222=={}".format(att_max))
    att_max[att_max == 0.] = 1.
    #print("att_min1111=={}".format(att_min))
    #print("data=={}".format(data.size()))
    #print("mask=={}".format(mask.size()))
    
    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!") 

    if torch.isnan(data_norm).any():
        raise Exception("nans!")
    """mask1 = mask.clone()
    for dx in range(len.shape[0]):
         mask1[dx, int(len[dx,0]):int(len[dx,0])+2, :] = 1"""
    # set masked out elements back to zero
    data_norm[mask == 0] = 0
    #print("data_norm3333=={}".format(data_norm))
    #print("data_norm44444=={}".format(data_norm.size()))
    return data_norm, att_min, att_max

#获得条件矩阵
def get_condition(value, data_min, data_max):
    #device = 'cpu'
    ix = value.shape[0]
    iy = value.shape[1]
    #condition = torch.zeros([2, iy]).to(device)
    condition_1 = torch.min(value , dim=0)[0]
    condition_2 = torch.max(value , dim=0)[0]
    #condition_1[condition_1 == 0] = data_min[condition_1 == 0]
    #condition_2[condition_2 == 0] = data_max[condition_2 == 0]
    condition_1 = torch.unsqueeze(condition_1,0)#升维
    condition_2 = torch.unsqueeze(condition_2,0)
    #con_1 = [condition_1 , condition_2]
    #print("type=={}".format(condition_2))
    condition = torch.cat((condition_1,condition_2), 0)
    #print("condition1=={}".format(condition)) 
    return condition    

def data_collate_1(batch , device, data_min=None, data_max=None):
    #device = 'cpu'
    D = batch[0][2].shape[1]
    #print("DDDDDDD=={}".format(D))#7
    N = len(batch)  #样本总数
    max_len = np.max([len(record[1]) for record in batch]) + 2
    combined_time = torch.zeros([N, max_len]).to(device)
    combined_value = torch.zeros([N, max_len, D]).to(device)
    combined_mask = torch.zeros([N, max_len, D]).to(device)
    combined_len = torch.zeros([N, 1]).to(device)
    combined_cond = torch.zeros([N, 2, D])
    combined_label = torch.zeros([N, 1]).to(device)
    for idx, (stay_id, time, vitalsign, mask, label) in enumerate(batch):
        """if idx == 1:
            print("time=={}".format(time))"""
        cur_len = time.shape[0]
        combined_len[idx , 0] = time.shape[0]
        combined_time[idx, :cur_len] = time.to(device)
        combined_value[idx,:cur_len, :D] = vitalsign.to(device)
        combined_mask[idx,:cur_len, :D] = mask.to(device)
        combined_cond[idx,0:2, :D]  = get_condition(vitalsign, data_min, data_max).to(device)
        combined_label[idx,0] = label.to(device) 
        assert combined_mask[idx].sum(-1).sum(-1) > 0, (idx, stay_id, time, vitalsign, mask)
    raw_value = combined_value
    combined_value, _, _ = normalize_masked_data_2(combined_value, combined_mask, att_min=data_min.to(device), att_max=data_max.to(device))
    
    """print("combined_time11111=={}".format(combined_time))
    combined_time1 = combined_time.cpu().numpy()
    np.savetxt('./zeros_array.csv', combined_time1, delimiter=',')"""
    if torch.max(combined_time) != 0:
       #print("max=={}".format(torch.max(combined_time)))
       combined_time = combined_time / torch.max(combined_time)

    #print("combined_time222222=={}".format(combined_time))
    combined_data = combined_value 
    return combined_data, combined_mask, combined_len, raw_value, combined_time, combined_cond

def data_collate_mimic(batch , device, data_min=None, data_max=None):
    #device = 'cpu'
    D = batch[0][2].shape[1]
    #print("DDDDDDD=={}".format(D))#7
    N = len(batch)  #样本总数
    max_len = np.max([len(record[1]) for record in batch]) + 2
    combined_time = torch.zeros([N, max_len]).to(device)
    combined_value = torch.zeros([N, max_len, D]).to(device)
    combined_mask = torch.zeros([N, max_len, D]).to(device)
    combined_len = torch.zeros([N, 1]).to(device)
    combined_cond = torch.zeros([N, 2, D])
    for idx, (time, vitalsign, mask) in enumerate(batch):
        """if idx == 1:
            print("time=={}".format(time))"""
        cur_len = time.shape[0]
        combined_len[idx , 0] = time.shape[0]
        combined_time[idx, :cur_len] = time.to(device)
        combined_value[idx,:cur_len, :D] = vitalsign.to(device)
        combined_mask[idx,:cur_len, :D] = mask.to(device)
        combined_cond[idx,0:2, :D]  = get_condition(vitalsign, data_min, data_max).to(device)
        #print("cond=={}".format(cond))
        #combined_value[idx,cur_len:cur_len+2, :D] = cond
        #combined_value[idx,cur_len:cur_len+2, :D] = 1
        #assert combined_mask[idx].sum(-1).sum(-1) > 0, (idx, stay_id, time, vitalsign, mask)
    raw_value = combined_value
    combined_value, _, _ = normalize_masked_data_2(combined_value, combined_mask,att_min=data_min.to(device), att_max=data_max.to(device))
    #raw_value = combined_value
    """print("combined_time11111=={}".format(combined_time))
    combined_time1 = combined_time.cpu().numpy()
    np.savetxt('./zeros_array.csv', combined_time1, delimiter=',')"""
    if torch.max(combined_time) != 0:
       #print("max=={}".format(torch.max(combined_time)))
       combined_time = combined_time / torch.max(combined_time)

    #print("combined_time222222=={}".format(combined_time))
    combined_data = combined_value 
    return combined_data, combined_mask, combined_len, raw_value, combined_time, combined_cond
    
def data_collate_fn(batch, device, data_min=None, data_max=None):
    '''
    time normalization, padding 
    batch: N records, for each record:(stay_id, time, vitalsign, mask, label)
    time: (T,)
    vitalsign: (T, D)
    mask: (T, D)
    label: float

    RETURN:
    combined_time: (N, T') --> (N, T', 1), T' = max(T)
    combined_vitalsign: (N, T', D)
    combined_mask: (N, T', D)
    combined_data: (N, T', D+D+1)
    combined_labe: (N, 1)
    '''
    device = 'cpu'
    D = batch[0][2].shape[1]
    #print("DDDDDDD=={}".format(D))#7
    N = len(batch)
    max_len = np.max([len(record[1]) for record in batch])
    #print("max_len=={}".format(max_len)) 200
    combined_time = torch.zeros([N, max_len]).to(device)
    combined_value = torch.zeros([N, max_len, D]).to(device)
    #print("combined_value=={}".format(combined_value.shape))#(1621,200,7)
    combined_mask = torch.zeros([N, max_len, D]).to(device)
    combined_label = torch.zeros([N, 1]).to(device)
    combined_len = torch.zeros([N, 1]).to(device)

    for idx, (stay_id, time, vitalsign, mask, label) in enumerate(batch):
        if idx == 1:
            print("timeshape=={}".format(time.shape[0]))
        cur_len = time.shape[0]
        
        combined_len[idx , 0] = time.shape[0]
        combined_time[idx,:cur_len] = time.to(device)
        #print("device=={}".format(device))
        #print("vitalsign=={}".format(vitalsign))
        combined_value[idx,:cur_len, :D] = vitalsign.to(device)
        combined_mask[idx,:cur_len, :D] = mask.to(device)
        combined_label[idx,0] = label.to(device) 
        assert combined_mask[idx].sum(-1).sum(-1) > 0, (idx, stay_id, time, vitalsign, mask)
    
    #print("combined_time_size111=={}".format(combined_time.shape))
    combined_value, _, _ = normalize_masked_data_2(combined_value, combined_mask, att_min=data_min.to(device), att_max=data_max.to(device))
    if torch.max(combined_time) != 0:
        combined_time = combined_time / torch.max(combined_time)
    #print("combined_value=={}".format(combined_value.shape))
    #print("combined_time=={}".format(combined_time.shape))
    combined_data = combined_value 
    
    """combined_data = torch.cat(
        (combined_value, combined_time.unsqueeze(-1)), 2)"""
    #print("combined_data=={}".format(combined_data))
    #print("combined_data_size=={}".format(combined_data.shape))
    #print("combined_time_size=={}".format(combined_time.shape))
    return combined_data, combined_label


def _load_data(data_path, vitalsign_list, label_0_indexd = True):
    """
    将数据文件转换成模型输入格式： (id, time, vitalsign, mask, label)
    vitalsign_list: 需要加入模型的生命体征
    默认使用格拉斯评分、体温
    """
    count_dict = {}
    raw_data = pd.read_csv(data_path)
    data = []
    for i, row in raw_data.iterrows():
        row_list = []
        row_list.append(int(row['PVID']))
        # time,vitalsign都是用逗号分隔的字符串
        if ',' in row['时序时间间隔']:
            time = [int(x) for x in json.loads(row['时序时间间隔'])]
            vs_list = [json.loads(row[col].replace("nan", "NaN")) for col in vitalsign_list]
        else:
            time = [int(x) for x in row['时序时间间隔'].strip('[]').split()]#strip字符串开头结尾删除字符，默认空格
            vs_list = []
            for col in vitalsign_list:
                vs_list.append([x for x in row[col].strip('[]').split()])#split()指定分隔符切片
        row_list.append(torch.Tensor(time))
        #print("vslist=={}".format(vs_list))
        #print("vslist=={}".format(type(vs_list)))
        vitalsign = np.array(vs_list, dtype=np.float64).T
        seq_len = vitalsign.shape[0]
        if seq_len != len(time) or seq_len > 200:
            continue
        vitalsign = np.concatenate([vitalsign, np.full((seq_len, 1), row['格拉斯评分']), np.full((seq_len, 1), row['体温'])], axis=1)
        """if i == 0:
            print("vitalsign=={}".format(vitalsign))"""
        mask = (~np.isnan(vitalsign)).astype(int)
        row_list.append(torch.Tensor(pd.DataFrame(data=vitalsign).fillna(0).values))
  
        row_list.append(torch.Tensor(mask))
       
        if not label_0_indexd:
            row['Label'] -= 1
        row_list.append(torch.tensor(row['Label']))
        #print("row_list=={}".format(row_list))
        data.append(tuple(row_list))
        """if i == 0:
            print("data=={}".format(data))"""
        if row['Label'] not in count_dict:
            count_dict[row['Label']]=1
        else:
            count_dict[row['Label']]+=1
    print("label count: ", count_dict)
    #print("data=={}".format(data[0]))
    print("type(data)=={}".format(type(data)))
    return data


def _load_mimic(data_path, vitalsign_list, label_0_indexd = True):
    """
    将数据文件转换成模型输入格式： (id, time, vitalsign, mask, label)
    vitalsign_list: 需要加入模型的生命体征
    默认使用格拉斯评分、体温
    """
    raw_data = pd.read_csv(data_path)
    data = []
    for i, row in raw_data.iterrows():
        row_list = []
        # time,vitalsign都是用逗号分隔的字符串
        if ',' in row['time']:
            time = [int(x) for x in json.loads(row['time'])]
            vs_list = [json.loads(row[col].replace("nan", "NaN")) for col in vitalsign_list]
        else:
            time = [int(x) for x in row['time'].strip('[]').split()]#strip字符串开头结尾删除字符，默认空格
            vs_list = []
            for col in vitalsign_list:
                vs_list.append([x for x in row[col].strip('[]').split()])#split()指定分隔符切片
        row_list.append(torch.Tensor(time))
        #print("vslist=={}".format(vs_list))
        #print("vslist=={}".format(type(vs_list)))
        vitalsign = np.array(vs_list, dtype=np.float64).T
        seq_len = vitalsign.shape[0]
        if seq_len != len(time) or seq_len > 200:
            continue
        mask = (~np.isnan(vitalsign)).astype(int)
        row_list.append(torch.Tensor(pd.DataFrame(data=vitalsign).fillna(0).values))
        row_list.append(torch.Tensor(mask))
        data.append(tuple(row_list))
    return data

def get_data(data_path, args, device):
    mimic_list = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
    vitalsign_list = ['心率', '呼吸率', '收缩压', '舒张压', '氧饱和度']
    healthcare_list = ['NIDiasABP', 'NISysABP', 'RespRate', 'Temp', 'NIMAP', 'HR']
    # dataset = torch.load(data_path, map_location='cpu')
    if args.dataset == '301':
       dataset = _load_data(data_path, vitalsign_list)
       data_min, data_max = get_data_min_max(dataset, device='cpu')
    elif args.dataset == 'mimic':
       dataset = _load_mimic(data_path, mimic_list)
       data_min, data_max = get_mimic_min_max(dataset, device='cpu')
    elif args.dataset == 'phythonet':
       dataset = _load_mimic(data_path, healthcare_list)
       data_min, data_max = get_mimic_min_max(dataset, device='cpu')
    #print("dataset=={}".format(len(dataset)))
    #print("dataset=={}".format(dataset[0]))
    label = [record[-1].numpy() for record in dataset]
    

    # 手动构造五折，记录五折结果
    #label = [record[-1].numpy() for record in dataset]
    data2, data1 = model_selection.train_test_split(dataset,train_size=0.8,random_state=0,shuffle=True)
    #print("dataset1=={}".format(len(data1)))
    #print("dataset2=={}".format(len(data2)))
    #label = [record[-1].numpy() for record in data2]
    data3, data2 = model_selection.train_test_split(data2,train_size=0.75,random_state=0,shuffle=True)
    #label = [record[-1].numpy() for record in data3]
    data4, data3 = model_selection.train_test_split(data3,train_size=0.666667,random_state=0,shuffle=True)
    #label = [record[-1].numpy() for record in data4]
    data5, data4 = model_selection.train_test_split(data4,train_size=0.5,random_state=0,shuffle=True)
    #print("dataset1=={}".format(len(data4)))
    #print("dataset2=={}".format(len(data5)))
    data = [data1, data2, data3, data4, data5]
    """for i in range(5):
        print("|||||||||||||||||||||||||||||||||")
        print(type(data))
        print(len(data[i]))"""
    batch_size = min(len(dataset), args.batch_size)
    print("|||||||||||||||||||||||||||||||||")
    print(data_min, data_max)
    time, vitalsign, mask = data1[0]

    train_dataloader_list, test_dataloader_list = [], []

    n_fold = 1
    for i in range(n_fold):
        # test_data_combined = data_collate_fn(data[i], device, torch.Tensor(fix_data_min[attr_list].values), torch.Tensor(fix_data_max[attr_list].values))
        
        #print("test_data_combined=={}".format(test_data_combined[0].shape))
        train_data = []
        for j in range(5):
            if i==j:
                continue
            train_data += data[j]
        # train_data_combined = data_collate_fn(train_data, device, torch.Tensor(fix_data_min[attr_list].values), torch.Tensor(fix_data_max[attr_list].values))
        if args.dataset == '301':
            test_data_combined = data_collate_1(data[i], device, data_min, data_max)
            train_data_combined = data_collate_1(train_data, device, data_min, data_max)
        elif args.dataset == 'mimic':
            test_data_combined = data_collate_mimic(data[i], device, data_min, data_max)
            train_data_combined = data_collate_mimic(train_data, device, data_min, data_max)
        elif args.dataset == 'phythonet':
            test_data_combined = data_collate_mimic(data[i], device, data_min, data_max)
            train_data_combined = data_collate_mimic(train_data, device, data_min, data_max)
        """print("len=={}".format(len(test_data_combined)))
        print("len1=={}".format(train_data_combined[0].shape))
        print("len2=={}".format(test_data_combined[1].shape))
        print("len3=={}".format(test_data_combined[2].size(0)))
        print("len4=={}".format(test_data_combined[3].shape))"""
        train_dataset = TensorDataset(train_data_combined[0], train_data_combined[1], train_data_combined[2], train_data_combined[3], train_data_combined[4], train_data_combined[5])
        test_dataset = TensorDataset(test_data_combined[0], test_data_combined[1],test_data_combined[2], test_data_combined[3], test_data_combined[4], test_data_combined[5])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_dataloader_list.append(train_dataloader)
        test_dataloader_list.append(test_dataloader)
    print("////////////////////////////////////")
    print(len(train_dataloader_list))
    print(len(test_dataloader_list[0]))
    data_object = {'dataset_obj': dataset,
                    'train_dataloader_list': train_dataloader_list,
                    # 'val_dataloader': val_dataloader,
                    'test_dataloader_list': test_dataloader_list,
                    'input_dim': vitalsign.shape[-1],
                    'n_train_batch': len(train_dataloader_list[0]),
                    'n_test_batch': len(test_dataloader_list[0]),
                    'attr': mimic_list,
                    'n_labels': 4
                    }
    return data_object, data_min, data_max