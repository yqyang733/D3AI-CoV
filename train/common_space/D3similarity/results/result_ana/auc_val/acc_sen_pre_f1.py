import sys, csv

def read_file(file):
    lab_pre = []
    with open(file) as f:
        f1 = csv.reader(f)
        for line in f1:
            lab_pre.append([line[1],line[0]])
    return lab_pre

def get_nums(lis):
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in lis:
        if float(i[0]) == 1 and float(i[1]) >= 0.5:
            TP += 1
        if float(i[0]) == 1 and float(i[1]) < 0.5:
            FN += 1
        if float(i[0]) == 0 and float(i[1]) >= 0.5:
            FP += 1
        if float(i[0]) == 0 and float(i[1]) < 0.5:
            TN += 1
    return TP, FP, FN, TN

def get_acc(lis):
    TP, FP, FN, TN = get_nums(lis)
    acc = (TP + TN)/(TP + FP + FN + TN)
    return acc

def get_pre(lis):
    TP, FP, FN, TN = get_nums(lis)
    pre = TP/(TP + FP)
    return pre

def get_sen(lis):
    TP, FP, FN, TN = get_nums(lis)
    sen = TP/(TP + FN)
    return sen

def get_f1(lis):
    sen = get_sen(lis)
    pre = get_pre(lis)
    f1 = 2*sen*pre/(sen + pre)
    return f1

def process_1file(file):
    lis = read_file(file)
    acc = get_acc(lis)
    pre = get_pre(lis)
    sen = get_sen(lis)
    f1 = get_f1(lis)
    return acc, pre, sen, f1

def all_file():
    output = "acc,pre,sen,f1\n"
    for i in range(10):
        acc, pre, sen, f1 = process_1file("pre_lab_" + str(i) + ".csv")
        output = output + str(acc) + "," + str(pre) + "," + str(sen) + "," + str(f1) + "\n"
    with open("val_acc_pre_sen_f1.csv", "w") as f:
        f.write(output)

def main():
    all_file()

if __name__ == '__main__':
    main()