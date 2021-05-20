import torch
import numpy as np
import pandas as pd

from config import Config
from utils.util import Helper
from model.CommonModel import Common_model
from dataset import HetrDataset

#导入模型参数
com_parm = torch.load('./results/com_model_parm/repeat_0_corss_0.parm')
#print(com_parm)

config = Config()
helper = Helper()
com_model = Common_model(config)
dataset = HetrDataset()

this_use_gpu = False
smi_input = dataset.smi_input
fas_input = dataset.fas_input
smi_input = helper.to_longtensor(smi_input,this_use_gpu)
fas_input = helper.to_longtensor(fas_input,this_use_gpu)

com_model.load_state_dict(com_parm)
com_model.eval()

#将异构数据集的公共空间保存起来
#模型训练完成后，他们的表示都是固定的

#获取drug在公共空间的表示
batch = 10
drug_com = None
sindex = 0
eindex = batch
with torch.no_grad():
    while eindex < len(smi_input):
        smi_common, _= com_model(smi_input[sindex:eindex], fas_input[0:batch])
        smi_common = smi_common.numpy()
        if sindex == 0:
            drug_com = smi_common
        else:
            drug_com = np.concatenate((drug_com, smi_common), axis=0)

        temp = eindex
        eindex = eindex + batch
        sindex = temp

    if eindex >= len(smi_input):
        new_batch = len(smi_input) - sindex
        smi_common, _ = com_model(smi_input[sindex:], fas_input[0:new_batch])
        smi_common = smi_common.numpy()
        print("drug_com:",drug_com)
        print("smi_common:",smi_common)
        drug_com = np.concatenate((drug_com, smi_common), axis=0)
print(len(drug_com))

#获取target在公共空间的表示
batch = 10
target_com = None
sindex = 0
eindex = batch
with torch.no_grad():
    while eindex < len(fas_input):
        _, fas_common = com_model(smi_input[0:batch], fas_input[sindex:eindex])
        fas_common = fas_common.numpy()
        if sindex == 0:
            target_com = fas_common
        else:
            target_com = np.concatenate((target_com, fas_common), axis=0)

        temp = eindex
        eindex = eindex + batch
        sindex = temp

    if eindex >= len(fas_input):
        new_batch = len(fas_input) - sindex
        _, fas_common= com_model(smi_input[0:new_batch], fas_input[sindex:])
        fas_common = fas_common.numpy()
        target_com = np.concatenate((target_com, fas_common), axis=0)
print(len(target_com))

#因为公共空间的获取需要一定时间，所有这里各个节点在公共空间的表示存储起来
dg_pt = pd.read_csv('data/drug_protein.csv',header=0,index_col=0)
dg_ds = pd.read_csv('data/drug_disease.csv', header=0, index_col=0)
dg_se = pd.read_csv('data/drug_se.csv', header=0, index_col=0)

dg_file = pd.DataFrame(drug_com,index=dg_pt._stat_axis.values)
dg_file.to_csv('results/drug_com.csv')

pt_file = pd.DataFrame(target_com,index=dg_pt.columns.values)
pt_file.to_csv('results/target_com.csv')