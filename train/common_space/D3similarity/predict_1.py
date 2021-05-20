import os, sys
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

#导入异构数据集在公共空间的表示,并转换为tensor格式
target_com = pd.read_csv('results/target_com.csv',index_col=0,header=0)
#print(target_com)
target_index = target_com._stat_axis.values
target_com = torch.FloatTensor(target_com.values)
#print(target_com)

#导入模型参数
com_parm = torch.load('./results/com_model_parm/repeat_0_corss_4.parm')
pre_parm = torch.load('./results/pre_model_parm/repeat_0_corss_4.parm')
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

def transmol_linux(in_name,out_name):
    in_suffix=os.path.splitext(in_name)[1][1:]
    out_suffix=os.path.splitext(out_name)[1][1:]
    if out_suffix == "smi":
        command="obabel -i%s %s -ocan -O %s"%(in_suffix,in_name,out_name)
    else:
        command="obabel -i%s %s -o%s -O %s"%(in_suffix,in_name,out_suffix,out_name)
    os.system(command)

def seq2num(smi):
    #将smiles序列通过字典的映射，转为数值表示
    output = np.zeros(config.smiles_max_len, dtype=int)

    for i in range(len(smi) - config.smi_n_gram + 1):
        key = smi[i:i + config.smi_n_gram]
        output[i] = smi_dict[key]
    return output

def predict_(smiless):
    temp = seq2num(smiless)
    input = []
    input.append(temp)
    input.append(temp)
    input = np.array(input)
    input = torch.LongTensor(input)
    with torch.no_grad():
        input_common, _= com_model(input,fas_input[0:2])
    tag = 1       #这个可以随便设定
    target_predict = dict()
    with torch.no_grad():
        for i in range(len(target_com)):
            predict_rate,tag = pre_model(input_common,target_com[i],tag)  #predict_rate就表示他们之间有相互作用的可能性大小
            target_predict[target_index[i]] = predict_rate[0].item()      #针对单个数据，将tensor转为python数据类型，即把tensor去掉
    return target_predict

# #异构网络的节点
# temp_name = "DB00385"
# temp = "c1cc(c2c(c1)C(=O)c1c(C2=O)c(O)c2c(c1O)C[C@](C[C@@H]2O[C@H]1C[C@@H]([C@@H]([C@@H](O1)C)O)NC(=O)C(F)(F)F)(C(=O)COC(=O)CCCC)O)OC"

# temp = seq2num(temp)
# input = []
# input.append(temp)
# input.append(temp)
# input = np.array(input)
# input = torch.LongTensor(input)
# #print(input)
# #得到这个药物在公共空间的表示，不管是原本就在公共空间内的，还是不在公共空间内的drug
# with torch.no_grad():
#     input_common, _= com_model(input,fas_input[0:2])
#     #print(input_common)

# #进行靶标预测,因为输入需要2个样本，所以这里就直接将input_common直接输入，得到2个相同的结果
# tag = 1       #这个可以随便设定
# target_predict = dict()
# with torch.no_grad():

#     for i in range(len(target_com)):
#         predict_rate,tag = pre_model(input_common,target_com[i],tag)  #predict_rate就表示他们之间有相互作用的可能性大小
#         target_predict[target_index[i]] = predict_rate[0].item()      #针对单个数据，将tensor转为python数据类型，即把tensor去掉

#     # print(target_predict)
#     # sort_predict = sorted(target_predict.items(),key= lambda x:x[1],reverse=True)   #reverse默认为False，按从小到大排序
#     # print(sort_predict)

def read_all_target():
    with open("proname_uniprot.txt") as f:
        f1 = f.readlines()
    uni_pro = {}
    for line in f1:
        uni_pro[line.split(",")[0]] = line.strip().split(",")[1]
    return uni_pro

def result_process(mol_name, uni_pro, target_predict_dict):
    output = "drug,target,possibility\n"
    for key in target_predict_dict:
        output = output + mol_name + "," + uni_pro[key] + "," + str(target_predict_dict[key]) + "\n"
    if not os.path.exists('./result_output'):
        os.mkdir('./result_output')
    with open('./result_output' + "/target_predict.txt","w") as rt:
        rt.write(output)

def main():
    mol_file = str(sys.argv[1])
    mol_name = os.path.splitext(mol_file)[0]
    transmol_linux(mol_file, mol_name + ".smi")
    with open(mol_name + ".smi") as f:
        smiles = f.readline().strip().split("\t")[0]
    # target_name, target_fasta = read_target()
    # target_predict(smiles, mol_name, target_name, target_fasta)
    print(smiles)
    target_predict_dict = predict_(smiles)
    uni_pro = read_all_target()
    result_process(mol_name, uni_pro, target_predict_dict)

if __name__=="__main__":
    main()
