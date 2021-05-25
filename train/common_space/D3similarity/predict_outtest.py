import os, sys
import torch
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

from config import Config
from utils.util import Helper
from model.CommonModel import Common_model
from model.PredictModel import Predict_model
from dataset import HetrDataset


"""
这个文件的功能：
靶标预测及虚拟筛选
"""

def smiles_read(smiles):
    with open(smiles) as f:
        f1 = f.readlines()
    dict_icv_drug = {}
    for line in f1:
        dict_icv_drug[line.split(",")[0]] = line.split(",")[1].strip()
        #break
    #print(dict_icv_drug_encoding)
    return dict_icv_drug

def protein_read(protein):
    with open(protein) as f:
        f1 = f.readlines()
    dict_target_protein = {}
    for line in f1:
        dict_target_protein[line.split(",")[0]] = line.split(",")[1].strip()
    # print(len(dict_target_protein_encoding["P0C6X7_3C"]))
    # print(len(f1[0]))
    return dict_target_protein

def data_process():
    out_dict_drug = smiles_read("./outtest/drug_smiles.txt")
    out_dict_protein = protein_read("./outtest/protein_fasta.txt")
    target_set = list(out_dict_protein.keys())
    with open("./outtest/drug_protein.txt") as f:
        f1 = f.readlines()
    dti_mutidict = defaultdict(list)
    [dti_mutidict[i.split(",")[0]].append(i.split(",")[1].strip()) for i in f1]
    whole_positive = []
    whole_negetive = []
    for key in dti_mutidict:
        for i in dti_mutidict[key]:
            whole_positive.append([key,i,out_dict_drug[key],out_dict_protein[i],1])
        target_no = target_set[:]
        [target_no.remove(i) for i in dti_mutidict[key]]
        for a in target_no:
            whole_negetive.append([key,a,out_dict_drug[key],out_dict_protein[a],0])
    whole_positive = np.array(whole_positive,dtype=object)
    whole_negetive = np.array(whole_negetive,dtype=object)
    data_set = np.vstack((whole_positive,whole_negetive))
    return data_set

with open('./data/smi_dict.pickle',"rb") as f:
    smi_dict = pickle.load(f)
with open('data/fas_dict.pickle',"rb") as f:
    fas_dict = pickle.load(f)

def load_mod(num):
    #导入字典

    #导入异构数据集在公共空间的表示,并转换为tensor格式
    target_com = pd.read_csv('./results/result_' + num + '/target_com_' + num + '.csv',index_col=0,header=0)
    #print(target_com)
    target_index = target_com._stat_axis.values
    target_com = torch.FloatTensor(target_com.values)
    #print(target_com)

    #导入模型参数
    com_parm = torch.load('./results/com_model_parm/repeat_0_corss_' + num + '.parm')
    pre_parm = torch.load('./results/pre_model_parm/repeat_0_corss_' + num + '.parm')
    #print(com_parm)
    #print(pre_parm)
    return com_parm, pre_parm

# com_parm, pre_parm = load_mod(num)
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

# com_model.load_state_dict(com_parm)
# pre_model.load_state_dict(pre_parm)
# com_model.eval()
# pre_model.eval()

def transmol_linux(in_name,out_name):
    in_suffix=os.path.splitext(in_name)[1][1:]
    out_suffix=os.path.splitext(out_name)[1][1:]
    if out_suffix == "smi":
        command="obabel -i%s %s -ocan -O %s"%(in_suffix,in_name,out_name)
    else:
        command="obabel -i%s %s -o%s -O %s"%(in_suffix,in_name,out_suffix,out_name)
    os.system(command)

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

def pair_predict(drug_name,drug_smi,target_name,target_fas,tag):
    drug_smi = dg_seq2num(drug_smi)
    dg_input = []
    dg_input.append(drug_smi)
    dg_input.append(drug_smi)
    dg_input = np.array(dg_input)
    dg_input = torch.LongTensor(dg_input)

    target_fas = tg_seq2sum(target_fas)
    tg_input = []
    tg_input.append(target_fas)
    tg_input.append(target_fas)
    tg_input = np.array(tg_input)
    tg_input = torch.LongTensor(tg_input)

    with torch.no_grad():
        dg_common, tg_common= com_model(dg_input,tg_input)
        predict_rate,tag_p = pre_model(dg_common,tg_common,tag)
        # print(drug_name,target_name,tag,predict_rate[0].item())
    return drug_name, target_name, predict_rate[0].item(), tag

def final_process(num):
    out_put = "drug_name,uniprot,y_pre,y_label\n"
    out_set = data_process()
    for i in out_set:
        # print(i[0],i[2],i[1],i[3],i[4])
        drug_name, target_name, predic, label = pair_predict(i[0],i[2],i[1],i[3],i[4])
        out_put = out_put + drug_name + "," + target_name + "," + str(predic) + "," + str(label) + "\n"
    with open("./results/result_" + num + "/outtest_pre_lab.csv", "w") as f:
        f.write(out_put)

def main():
    num = str(sys.argv[1])
    # out_set = data_process()
    com_parm, pre_parm = load_mod(num)
    com_model.load_state_dict(com_parm)
    pre_model.load_state_dict(pre_parm)
    com_model.eval()
    pre_model.eval()
    final_process(num)


if __name__=="__main__":
    main()
