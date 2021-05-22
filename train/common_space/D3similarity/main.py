import time
import datetime
import os

import numpy as np
import torch.optim as optim
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from config import Config
from utils.util import Helper
from model.CommonModel import Common_model
from model.PredictModel import Predict_model
from dataset import HetrDataset

def train_common_model(config,helper,model,hetrdataset,repeat_nums,flod_nums):

    optimizer = optim.Adam(model.parameters(),config.common_learn_rate)
    model.train()

    print("common model begin training----------",datetime.datetime.now())

    #common_loss
    for e in range(config.common_epochs):

        common_loss = 0
        begin_time = time.time()

        for i, (dg,pt,tag) in enumerate(hetrdataset.get_train_batch(repeat_nums,flod_nums,config.batch_size)):
            dg = helper.to_longtensor(dg,config.use_gpu)
            pt = helper.to_longtensor(pt,config.use_gpu)
            tag = helper.to_floattensor(tag,config.use_gpu)

            optimizer.zero_grad()

            smi_common, fas_common = model(dg,pt)

            distance_loss = helper.comput_distance_loss(smi_common,fas_common,tag)
            common_loss += distance_loss

            distance_loss.backward()
            optimizer.step()

        #end a epech
        print("the loss of common model epoch[%d / %d]:is %4.f, time:%d s" % (e+1,config.common_epochs,common_loss,time.time()-begin_time))
    return common_loss.item()

def train_predict_model(config,helper,predict_model,common_model,hetrdataset,repeat_nums,flod_nums):

    optimizer1 = optim.Adam(predict_model.parameters(),config.pre_learn_rate)
    optimizer2 = optim.Adam(common_model.parameters(),config.common_learn_rate)

    predict_model.train()
    common_model.train()

    print("predict model begin training----------",datetime.datetime.now())

    #tag_loss
    for e in range(config.predict_epochs):

        epoch_loss = 0
        begin_time = time.time()

        for i, (dg,pt,tag) in enumerate(hetrdataset.get_train_batch(repeat_nums,flod_nums,config.batch_size)):
            dg = helper.to_longtensor(dg,config.use_gpu)
            pt = helper.to_longtensor(pt,config.use_gpu)
            tag = helper.to_floattensor(tag,config.use_gpu)

            smi_common,fas_common = common_model(dg,pt)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            predict, tag  = predict_model(smi_common,fas_common,tag)

            tag_loss = F.binary_cross_entropy(predict,tag)
            epoch_loss += tag_loss

            tag_loss.backward()

            optimizer1.step()
            optimizer2.step()

        # end a epech
        print("the loss of predict model epoch[%d / %d]:is %4.f, time:%d s" %  (e+1, config.predict_epochs, epoch_loss, time.time() - begin_time))

        #将每一折交叉验证的模型存储起来
        if not os.path.exists('./results'):
            os.mkdir('./results')
        if not os.path.exists('./results/com_model_parm'):
            os.mkdir('./results/com_model_parm')
        if not os.path.exists('./results/pre_model_parm'):
            os.mkdir('./results/pre_model_parm')

        #evaluation_model(config, helper, predict_model, common_model, hetrdataset, repeat_nums, flod_nums)
        if e == config.predict_epochs-1 and epoch == config.num_epochs-1:
            torch.save(common_model.state_dict(),
                       './results/com_model_parm/repeat_%d_corss_%d.parm' % (repeat_nums, flod_nums))
            torch.save(predict_model.state_dict(),
                       './results/pre_model_parm/repeat_%d_corss_%d.parm' % (repeat_nums, flod_nums))
            #评估模型
        loss, pre_all, lab_all = evaluation_model(config, helper, predict_model, common_model, hetrdataset, repeat_nums, flod_nums)

    return epoch_loss.item(), loss, pre_all, lab_all

def evaluation_model(config,helper,predict_model,common_model,hetrdataset,repeat_nums,flod_nums):
    predict_model.eval()
    common_model.eval()
    print("evaluate the model")

    begin_time = time.time()
    loss = 0
    avg_acc = []
    avg_aupr = []
    pre_all = []
    lab_all = []
    with torch.no_grad():
        for i,(dg,pt,tag) in enumerate(hetrdataset.get_test_batch(repeat_nums,flod_nums,config.batch_size)):
            dg = helper.to_longtensor(dg,config.use_gpu)
            pt = helper.to_longtensor(pt,config.use_gpu)
            tag = helper.to_floattensor(tag,config.use_gpu)

            smi_common,fas_common = common_model(dg,pt)
            predict, tag = predict_model(smi_common, fas_common, tag)

            pre_all.append(predict)
            lab_all.append(tag)

            tag_loss = F.binary_cross_entropy(predict,tag)
            loss +=tag_loss

            auc = roc_auc_score(tag.cpu(),predict.cpu())
            aupr = average_precision_score(tag.cpu(),predict.cpu())

            avg_acc.append(auc)
            avg_aupr.append(aupr)

    print("the total_loss of test model:is %4.f, time:%d s" % (loss, time.time() - begin_time))
    print("avg_acc:",np.mean(avg_acc),"avg_aupr:",np.mean(avg_aupr))

    return loss.item(), pre_all, lab_all

if __name__=='__main__':

    # initial parameters class
    config = Config()

    # initial utils class
    helper = Helper()

    #initial data
    hetrdataset = HetrDataset()

    #torch.backends.cudnn.enabled = False 把

    model_begin_time = time.time()
    for i in range(config.repeat_nums):
        print("repeat:",str(i),"+++++++++++++++++++++++++++++++++++")
        for j in range(config.fold_nums):
            print(" crossfold:", str(j), "----------------------------")
            if not os.path.exists('./results/result_' + str(j) + "/"):
                os.makedirs('./results/result_' + str(j) + "/")
            #initial presentation model
            c_model = Common_model(config)
            p_model = Predict_model()
            if config.use_gpu:
                c_model = c_model.cuda()
                p_model = p_model.cuda()

            loss_rt = "model_loss,predict_loss,val_loss\n"
            for epoch in range(config.num_epochs):
                print("         epoch:",str(epoch),"zzzzzzzzzzzzzzzz")
                common_loss = train_common_model(config,helper,c_model,hetrdataset,i,j)
                predict_loss, val_loss, pre_all, lab_all = train_predict_model(config,helper,p_model,c_model,hetrdataset,i,j)
                loss_rt = loss_rt + str(common_loss) + "," + str(predict_loss) + "," + str(val_loss) + "\n"
                print(str(common_loss) + "," + str(predict_loss) + "," + str(val_loss) + "\n")
            with open('./results/result_' + str(j) + "/" + "all_loss_" + str(j) + ".csv","w") as rt:
                rt.write(loss_rt)
            pre_lab = "y_pre,y_lab\n"
            # for i in range(len(pre_all)):
            #     pre_lab = pre_lab + str(pre_all[i]) + "," + str(lab_all[i]) + "\n"
            for qq in range(len(pre_all)):
                a = pre_all[qq].tolist()
                b = lab_all[qq].tolist()
                for m in range(len(a)):
                    pre_lab = pre_lab + str(a[m]) + "," + str(b[m]) + "\n"
            with open('./results/result_' + str(j) + "/" + "pre_lab_" + str(j) + ".csv","w") as rt:
                rt.write(pre_lab)

            # with open('./results/result_' + str(j) + "/" + "all_loss_" + str(j) + ".csv","w") as rt:
            #     rt.write("model_loss,predict_loss,val_loss\n")
            #     for epoch in range(config.num_epochs):
            #         print("         epoch:",str(epoch),"zzzzzzzzzzzzzzzz")
            #         common_loss = train_common_model(config,helper,c_model,hetrdataset,i,j)
            #         predict_loss, val_loss, pre_all, lab_all = train_predict_model(config,helper,p_model,c_model,hetrdataset,i,j)
            #         rt.write(str(common_loss) + "," + str(predict_loss) + "," + str(val_loss) + "\n")
            #         print(str(common_loss) + "," + str(predict_loss) + "," + str(val_loss) + "\n")
            #     pre_lab = "y_pre,y_lab\n"
            #     for i in range(len(pre_all)):
            #         a = pre_all[i].tolist()
            #         b = lab_all[i].tolist()
            #         for m in range(len(a)):
            #             pre_lab = pre_lab + str(a[m]) + "," + str(b[m]) + "\n"
            #     with open('./results/result_' + str(j) + "/" + "pre_lab_" + str(j) + ".csv","w") as rt:
            #         rt.write(pre_lab)


    print("Done!")
    print("All_training time:",time.time()-model_begin_time)