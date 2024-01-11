import torch
p = torch.tensor([1, 2, 3, 4, 5])
p1 = p
p[1,2] = 10

print(p)
<<<<<<< HEAD
print(p1)




for ix, (train_batch, mask, len, raw, time, cond) in enumerate(test_loader):         
            #if ix == 1: break
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            #print("train_batch=={}".format(train_batch))
            """combined_time1 = train_batch[1].cpu().numpy()
            np.savetxt('./12.csv', combined_time1, delimiter=',')"""
            cond_norm = tr_utils.normalize_cond(cond, data_min, data_max)
            train_batch = tr_utils.add_cond(train_batch , len , cond_norm)
            test_mask1,test_mask2 = tr_utils.set_testmask(mask,opt.mask_rate)
            mask1 = test_mask1.clone()
            for dx in range(len.shape[0]):
                mask1[dx, int(len[dx,0]):int(len[dx,0])+2, :] = 1
            train_batch = train_batch * mask1
            y_g2 = modelg1(train_batch)
            y3 = y_g2.detach().clone()
            y4 = y3.detach().clone()
            y_2 = tr_utils.replace_raw(y3,test_mask1, raw, len)
            y_2 = torch.cat((y_2, time.unsqueeze(-1)), 2)
            y_g3 = modelg2(y_2)
            y5 = y_g3.detach().clone()
            mae = tr_utils.caculate_mae(y5, test_mask2, len, raw)
            print("mae=={}".format(mae))
            mae_m = mae_m + mae
            mae_x = mae_x + 1
=======
print(p1)
>>>>>>> b7dfa4787de54de8d6d1b1cdd8d0a1f768a8f8ee
