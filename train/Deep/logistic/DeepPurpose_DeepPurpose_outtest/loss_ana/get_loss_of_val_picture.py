import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm,colors
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, show, rc

def col_pic(file):
    df = pd.read_csv(file)
    color = ["red","blue","black","green","maroon","purple","teal","orange","olive","skyblue"]
    fig = plt.figure(figsize=(10,8))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    ax = plt.gca()
    b1, = plt.plot(df['epoch'],df['fold1'],color=color[1],linewidth=2, label='Fold1')
    b2, = plt.plot(df['epoch'],df['fold2'],color=color[2],linewidth=2, label='Fold2')
    b3, = plt.plot(df['epoch'],df['fold3'],color=color[3],linewidth=2, label='Fold3')
    b4, = plt.plot(df['epoch'],df['fold4'],color=color[4],linewidth=2, label='Fold4')
    b5, = plt.plot(df['epoch'],df['fold5'],color=color[5],linewidth=2, label='Fold5')
    b6, = plt.plot(df['epoch'],df['fold6'],color=color[6],linewidth=2, label='Fold6')
    b7, = plt.plot(df['epoch'],df['fold7'],color=color[7],linewidth=2, label='Fold7')
    b8, = plt.plot(df['epoch'],df['fold8'],color=color[8],linewidth=2, label='Fold8')
    b9, = plt.plot(df['epoch'],df['fold9'],color=color[9],linewidth=2, label='Fold9')
    plt.legend(handles=[b1,b2,b3,b4,b5,b6,b7,b8,b9],loc=(0.57,0.85),ncol=3,frameon=False,prop="Times New Roman")
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=15, )
    plt.xlabel('Epoch', fontproperties="Times New Roman",fontsize=20,weight="bold")
    plt.ylabel('Cross-entropy Loss',fontproperties="Times New Roman",fontsize=20,weight="bold")
    plt.xticks(font="Times New Roman",size=15)
    plt.yticks(font="Times New Roman",size=15)
    fig.savefig('MPNN-CNN-Logistic-Val-loss.pdf')

def main():
    file = str(sys.argv[1])
    col_pic(file)
    
if __name__=="__main__":
    main() 