import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
import csv
import sys
import numpy as np

def ro_curve(y_pred, y_label, figure_file, method_name):
    '''
        y_pred is a list of length n.  (0,1)
        y_label is a list of same length. 0/1
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py  
    '''
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)    
    # fpr = dict()
    # tpr = dict() 
    # roc_auc = dict()
    # fpr[0], tpr[0], _ = precision_recall_curve(y_label, y_pred)
    # roc_auc[0] = auc(fpr[0], tpr[0])
    # lw = 2
    # plt.plot(fpr[0], tpr[0],
    #      lw=lw, label= method_name + ' (area = %0.2f)' % roc_auc[0])
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # fontsize = 14
    # plt.xlabel('Recall', fontsize = fontsize)
    # plt.ylabel('Precision', fontsize = fontsize)
    # plt.title('Precision Recall Curve')
    # plt.legend(loc="lower right")
    # plt.savefig(figure_file)
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
#   plt.plot([0,1], [no_skill, no_skill], linestyle='--')
    plt.plot(lr_recall, lr_precision, lw = 2, label= method_name + ' (area = %0.2f)' % average_precision_score(y_label, y_pred))
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.title('Precision Recall Curve')
    plt.legend()
    plt.savefig(figure_file)
    return 

def col_pic():
    for i in range(10):
        y_label = []
        y_pred = []
        with open("y_label_pred_" + str(i) + ".csv") as f:
            f1 = csv.reader(f)
            for line in f1:
                y_label.append(int(line[0]))
                # if float(line[1]) > 0.5:
                #     y_pred.append(1)
                # else:
                #     y_pred.append(0)
                y_pred.append(float(line[1]))
            ro_curve(y_pred,y_label,"aupr_test","Fold" + str(i+1))

def main():
    col_pic()
    
if __name__=="__main__":
    main() 