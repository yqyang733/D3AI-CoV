import torch
import numpy as np
import pandas as pd
import pickle

from config import Config
from utils.util import Helper
from model.CommonModel import Common_model
from model.PredictModel import Predict_model
from dataset import HetrDataset

"""
这个文件的功能：
靶标预测及虚拟筛选
"""

#导入字典
with open('data/smi_dict.pickle',"rb") as f:
    smi_dict = pickle.load(f)
with open('data/fas_dict.pickle',"rb") as f:
    fas_dict = pickle.load(f)

#导入异构数据集在公共空间的表示,并转换为tensor格式
target_com = pd.read_csv('results/target_com.csv',index_col=0,header=0)
#print(target_com)
target_index = target_com._stat_axis.values
target_com = torch.FloatTensor(target_com.values)
#print(target_com)

#导入模型参数
com_parm = torch.load('./results/com_model_parm/repeat_0_corss_0.parm')
pre_parm = torch.load('./results/pre_model_parm/repeat_0_corss_0.parm')
#print(com_parm)
#print(pre_parm)

config = Config()
helper = Helper()
com_model = Common_model(config)
pre_model = Predict_model()
dataset = HetrDataset()

this_use_gpu = False
smi_input = dataset.smi_input
fas_input = dataset.fas_input
smi_input = helper.to_longtensor(smi_input,this_use_gpu)
fas_input = helper.to_longtensor(fas_input,this_use_gpu)

com_model.load_state_dict(com_parm)
pre_model.load_state_dict(pre_parm)
com_model.eval()
pre_model.eval()

def dg_seq2num(smi):
    #将smiles序列通过字典的映射，转为数值表示
    output = np.zeros(config.smiles_max_len, dtype=int)

    for i in range(len(smi) - config.smi_n_gram + 1):
        key = smi[i:i + config.smi_n_gram]
        output[i] = smi_dict[key]
    return output

def tg_seq2sum(fas):
    # 将smiles序列通过字典的映射，转为数值表示
    output = np.zeros(config.fasta_max_len, dtype=int)

    for j in range(len(fas) - config.fas_n_gram + 1):
        key = fas[j:j + config.fas_n_gram]
        output[j] = fas_dict[key]
    return output

#功能1：靶标预测
drug_name = "DB00385"
drug_smi = "c1cc(c2c(c1)C(=O)c1c(C2=O)c(O)c2c(c1O)C[C@](C[C@@H]2O[C@H]1C[C@@H]([C@@H]([C@@H](O1)C)O)NC(=O)C(F)(F)F)(C(=O)COC(=O)CCCC)O)OC"
def target_predict(drug_name,drug_smi):
    drug_smi = dg_seq2num(drug_smi)
    dg_input = []
    dg_input.append(drug_smi)
    dg_input.append(drug_smi)
    dg_input = np.array(dg_input)
    dg_input = torch.LongTensor(dg_input)

    #得到这个药物在公共空间的表示，不管是原本就在公共空间内的，还是不在公共空间内的drug
    with torch.no_grad():
        dg_common, _ = com_model(dg_input, fas_input[0:2])

    # 进行靶标预测,因为输入需要2个样本，所以这里就直接将input_common直接输入，得到2个相同的结果
    tag = 1  # 这个可以随便设定
    target_predict = dict()
    with torch.no_grad():
        for i in range(len(target_com)):
            predict_rate, tag = pre_model(dg_common, target_com[i], tag)  # predict_rate就表示他们之间有相互作用的可能性大小
            target_predict[target_index[i]] = predict_rate[0].item()  # 针对单个数据，将tensor转为python数据类型，即把tensor去掉

        #print(target_predict)
        sort_predict = sorted(target_predict.items(), key=lambda x: x[1], reverse=True)  # reverse默认为False，按从小到大排序
        print('药物：',drug_name,'靶标预测结果（按照预测概率从大到小排列）：',sort_predict)
#测试
target_predict(drug_name,drug_smi)

#功能2：预测药物-靶标对
drug_name = "DB00385"
drug_smi = "c1cc(c2c(c1)C(=O)c1c(C2=O)c(O)c2c(c1O)C[C@](C[C@@H]2O[C@H]1C[C@@H]([C@@H]([C@@H](O1)C)O)NC(=O)C(F)(F)F)(C(=O)COC(=O)CCCC)O)OC"
target_name = ''
target_fas = ''
tag = 0   #或1
def pair_predict(drug_name,drug_smi,target_name,target_fas,tag):
    drug_smi = dg_seq2num(drug_smi)
    dg_input = []
    dg_input.append(drug_smi)
    dg_input.append(drug_smi)
    dg_input = np.array(dg_input)
    dg_input = torch.LongTensor(dg_input)

    target_fas = dg_seq2num(target_fas)
    tg_input = []
    tg_input.append(target_fas)
    tg_input.append(target_fas)
    tg_input = np.array(tg_input)
    tg_input = torch.LongTensor(tg_input)

    with torch.no_grad():
        dg_common, tg_common= com_model(dg_input,tg_input)
        predict_rate,tag = pre_model(dg_common,tg_common,tag)
        print(drug_name,target_name,tag,predict_rate[0].item())
#测试
pair_predict(drug_name,drug_smi,target_name,target_fas,tag)

