import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import csv

from os import listdir
from os.path import isfile, join

mypath = "GPL"
mypath2 = "QL"
mypath3 = "QL-AM"
mypath4 = "GNN"
mypath5 = "GNN-AM"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) if ".csv" in f]
onlyfiles2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f)) if ".csv" in f]
onlyfiles3 = [f for f in listdir(mypath3) if isfile(join(mypath3, f)) if ".csv" in f]
onlyfiles4 = [f for f in listdir(mypath4) if isfile(join(mypath4, f)) if ".csv" in f]
onlyfiles5 = [f for f in listdir(mypath5) if isfile(join(mypath5, f)) if ".csv" in f]

raw_data_1 = []
raw_data_2 = []
raw_data_3 = []
raw_data_4 = []
raw_data_5 = []


for file_res in onlyfiles:
    data1 = []
    with open(mypath+"/"+file_res) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        counter = 0
        for row in csv_reader:
            if counter != 0:
                data1.append(row[2])
            counter += 1
    raw_data_1.append(data1)

for file_res in onlyfiles2:
    data2 = []
    with open(mypath2+"/"+file_res) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        counter = 0
        for row in csv_reader:
            if counter != 0:
                data2.append(row[2])
            counter += 1
    raw_data_2.append(data2)

for file_res in onlyfiles3:
    data3 = []
    with open(mypath3+"/"+file_res) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        counter = 0
        for row in csv_reader:
            if counter != 0:
                data3.append(row[2])
            counter += 1
    raw_data_3.append(data3)

for file_res in onlyfiles4:
    data4 = []
    with open(mypath4+"/"+file_res) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        counter = 0
        for row in csv_reader:
            if counter != 0:
                data4.append(row[2])
            counter += 1
    raw_data_4.append(data4)

for file_res in onlyfiles5:
    data5 = []
    with open(mypath5+"/"+file_res) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        counter = 0
        for row in csv_reader:
            if counter != 0:
                data5.append(row[2])
            counter += 1
    raw_data_5.append(data5)

min_len1, min_len2, min_len3, min_len4, min_len5 = min([len(a) for a in raw_data_1]), min([len(a) for a in raw_data_2]), min([len(a) for a in raw_data_3]), min([len(a) for a in raw_data_4]), min([len(a) for a in raw_data_5])

min_all = min([min_len1, min_len2, min_len3, min_len4, min_len5])

raw_data_1 = [a[:min_all] for a in raw_data_1]
raw_data_2 = [a[:min_all] for a in raw_data_2]
raw_data_3 = [a[:min_all] for a in raw_data_3]
raw_data_4 = [a[:min_all] for a in raw_data_4]
raw_data_5 = [a[:min_all] for a in raw_data_5]

raw_data_np1 = np.asarray(raw_data_1).astype(np.float)
raw_data_np2 = np.asarray(raw_data_2).astype(np.float)
raw_data_np3 = np.asarray(raw_data_3).astype(np.float)
raw_data_np4 = np.asarray(raw_data_4).astype(np.float)
raw_data_np5 = np.asarray(raw_data_5).astype(np.float)

df1 = pd.DataFrame(raw_data_np1).melt()
df2 = pd.DataFrame(raw_data_np2).melt()
df3 = pd.DataFrame(raw_data_np3).melt()
df4 = pd.DataFrame(raw_data_np4).melt()
df5 = pd.DataFrame(raw_data_np5).melt()


plt.title("Average Shooting Accuracy in FortAttack during Testing")
df1 = df1.rename(columns={"variable": "Checkpoint", "value": "Average Shooting Accuracy per Episode"})
g = sns.lineplot(x="Checkpoint", y="Average Shooting Accuracy per Episode", data=df1, label="GPL-Q")
df2 = df2.rename(columns={"variable": "Checkpoint", "value": "Average Shooting Accuracy per Episode"})
g =sns.lineplot(x="Checkpoint", y="Average Shooting Accuracy per Episode", data=df2, label="QL")
df3 = df3.rename(columns={"variable": "Checkpoint", "value": "Average Shooting Accuracy per Episode"})
g =sns.lineplot(x="Checkpoint", y="Average Shooting Accuracy per Episode", data=df3, label="QL-AM")
df4 = df4.rename(columns={"variable": "Checkpoint", "value": "Average Shooting Accuracy per Episode"})
g =sns.lineplot(x="Checkpoint", y="Average Shooting Accuracy per Episode", data=df4, label="GNN")
df5 = df5.rename(columns={"variable": "Checkpoint", "value": "Average Shooting Accuracy per Episode"})
g =sns.lineplot(x="Checkpoint", y="Average Shooting Accuracy per Episode", data=df5, label="GNN-AM")
gd = g.legend(fontsize = 'x-small')
plt.savefig("FortAttackTrainingAccuracyPerBaseline.pdf", bbox_inches='tight')

