import os
import sys
f = open(sys.argv[1],'r')
lines = f.readlines()
acc_dict = dict()
for line in lines:
    line = line.strip()
    if 'Glocal_step' in line:
        step = int(line.split(':')[1])
    if 'local-' in line:
        line = line.replace(' ', '').strip('|')
        split = line.split('|')
        ID = int(split[0].split('-')[1])
        acc = [float(x) for x in split[1:]]
        if step == -1:
            acc_dict[ID] = []
        acc_dict[ID].append(acc)


improve_1e5 = []
improve_1e4 = []
list_1e5 = []
list_1e4 = []
pretrain_1e5 = []
pretrain_1e4 = []
for ID in acc_dict:
    e = 4
    print('ID %d: (1e-5) %.2f --> %.2f | (1e-4) %.2f --> %.2f'\
        %(ID,acc_dict[ID][0][1],acc_dict[ID][e][1],acc_dict[ID][0][2],acc_dict[ID][e][2]))
    improve_1e5.append(acc_dict[ID][e][1]-acc_dict[ID][0][1])
    improve_1e4.append(acc_dict[ID][e][2]-acc_dict[ID][0][2])
    list_1e5.append(acc_dict[ID][e][1])
    list_1e4.append(acc_dict[ID][e][2])
    pretrain_1e5.append(acc_dict[ID][0][1])
    pretrain_1e4.append(acc_dict[ID][0][2])
print('Improved : Avg 1e-5 : %.2f,   Avg 1e-4 : %.2f'\
    %(sum(improve_1e5)/len(improve_1e5),sum(improve_1e4)/len(improve_1e4)))
print('Pretrained : Avg 1e-5 : %.2f,   Avg 1e-4 : %.2f'\
    %(sum(pretrain_1e5)/len(pretrain_1e5),sum(pretrain_1e4)/len(pretrain_1e4)))
print('Trained : Avg 1e-5 : %.2f,   Avg 1e-4 : %.2f'\
    %(sum(list_1e5)/len(list_1e5),sum(list_1e4)/len(list_1e4)))