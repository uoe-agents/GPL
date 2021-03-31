import seaborn as sns; sns.set()
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import csv

from os import listdir
from os.path import isfile, join

mypath = "GNN"
mypath2 = "LSTM"
mypath3 = "LSTM-RFM"
mypath4 = "DGN-Only"
mypath5 = "DGN"
mypath6 = "MADDPG"
mypath7 = "DGN-MARL"
mypath8 = "GPL"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) if ".csv" in f]
onlyfiles2 = [f for f in listdir(mypath2) if isfile(join(mypath2, f)) if ".csv" in f]
onlyfiles3 = [f for f in listdir(mypath3) if isfile(join(mypath3, f)) if ".csv" in f]
onlyfiles4 = [f for f in listdir(mypath4) if isfile(join(mypath4, f)) if ".csv" in f]
onlyfiles5 = [f for f in listdir(mypath5) if isfile(join(mypath5, f)) if ".csv" in f]
onlyfiles6 = [f for f in listdir(mypath6) if isfile(join(mypath6, f)) if ".csv" in f]
onlyfiles7 = [f for f in listdir(mypath7) if isfile(join(mypath7, f)) if ".csv" in f]
onlyfiles8 = [f for f in listdir(mypath8) if isfile(join(mypath8, f)) if ".csv" in f]

raw_data_1 = []
raw_data_2 = []
raw_data_3 = []
raw_data_4 = []
raw_data_5 = []
raw_data_6 = []
raw_data_7 = []
raw_data_8 = []


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

for file_res in onlyfiles6:
    data6 = []
    with open(mypath6+"/"+file_res) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        counter = 0
        for row in csv_reader:
            if counter != 0:
                data6.append(row[2])
            counter += 1
    raw_data_6.append(data6)


for file_res in onlyfiles7:
    data7 = []
    with open(mypath7+"/"+file_res) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        counter = 0
        for row in csv_reader:
            if counter != 0:
                data7.append(row[2])
            counter += 1
    raw_data_7.append(data7)

for file_res in onlyfiles8:
    data8 = []
    with open(mypath8+"/"+file_res) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        counter = 0
        for row in csv_reader:
            if counter != 0:
                data8.append(row[2])
            counter += 1
    raw_data_8.append(data8)

min_len1, min_len2, min_len3, min_len4, min_len5 = min([len(a) for a in raw_data_1]), min([len(a) for a in raw_data_2]), min([len(a) for a in raw_data_3]), min([len(a) for a in raw_data_4]), min([len(a) for a in raw_data_5])
min_len6 = min([len(a) for a in raw_data_6])
min_len7 = min([len(a) for a in raw_data_7])
min_len8 = min([len(a) for a in raw_data_8])

min_all = min([min_len1, min_len2, min_len3, min_len4, min_len5, min_len6, min_len7, min_len8])

raw_data_1 = [a[:min_all] for a in raw_data_1]
raw_data_2 = [a[:min_all] for a in raw_data_2]
raw_data_3 = [a[:min_all] for a in raw_data_3]
raw_data_4 = [a[:min_all] for a in raw_data_4]
raw_data_5 = [a[:min_all] for a in raw_data_5]
raw_data_6 = [a[:min_all] for a in raw_data_6]
raw_data_7 = [a[:min_all] for a in raw_data_7]
raw_data_8 = [a[:min_all] for a in raw_data_8]

raw_data_np1 = np.asarray(raw_data_1).astype(np.float)
raw_data_np2 = np.asarray(raw_data_2).astype(np.float)
raw_data_np3 = np.asarray(raw_data_3).astype(np.float)
raw_data_np4 = np.asarray(raw_data_4).astype(np.float)
raw_data_np5 = np.asarray(raw_data_5).astype(np.float)
raw_data_np6 = np.asarray(raw_data_6).astype(np.float)
raw_data_np7 = np.asarray(raw_data_7).astype(np.float)
raw_data_np8 = np.asarray(raw_data_8).astype(np.float)

df1 = pd.DataFrame(raw_data_np1).melt()
df2 = pd.DataFrame(raw_data_np2).melt()
df3 = pd.DataFrame(raw_data_np3).melt()
df4 = pd.DataFrame(raw_data_np4).melt()
df5 = pd.DataFrame(raw_data_np5).melt()
df6 = pd.DataFrame(raw_data_np6).melt()
df7 = pd.DataFrame(raw_data_np7).melt()
df8 = pd.DataFrame(raw_data_np8).melt()

plt.title("Average Training Return in FortAttack")
df1 = df1.rename(columns={"variable": "Total Steps (x160000)", "value": "Average Return per Episode"})
g = sns.lineplot(x="Total Steps (x160000)", y="Average Return per Episode", data=df1, label="GPL-Q")
df8 = df8.rename(columns={"variable": "Total Steps (x160000)", "value": "Average Return per Episode"})
g =sns.lineplot(x="Total Steps (x160000)", y="Average Return per Episode", data=df8, label="GPL-SPI")
df2 = df2.rename(columns={"variable": "Total Steps (x160000)", "value": "Average Return per Episode"})
g =sns.lineplot(x="Total Steps (x160000)", y="Average Return per Episode", data=df2, label="QL")
df3 = df3.rename(columns={"variable": "Total Steps (x160000)", "value": "Average Return per Episode"})
g =sns.lineplot(x="Total Steps (x160000)", y="Average Return per Episode", data=df3, label="QL-AM")
df4 = df4.rename(columns={"variable": "Total Steps (x160000)", "value": "Average Return per Episode"})
g =sns.lineplot(x="Total Steps (x160000)", y="Average Return per Episode", data=df4, label="GNN")
df5 = df5.rename(columns={"variable": "Total Steps (x160000)", "value": "Average Return per Episode"})
g =sns.lineplot(x="Total Steps (x160000)", y="Average Return per Episode", data=df5, label="GNN-AM")
df6 = df6.rename(columns={"variable": "Total Steps (x160000)", "value": "Average Return per Episode"})
g =sns.lineplot(x="Total Steps (x160000)", y="Average Return per Episode", data=df6, label="MADDPG")
df7 = df7.rename(columns={"variable": "Total Steps (x160000)", "value": "Average Return per Episode"})
g =sns.lineplot(x="Total Steps (x160000)", y="Average Return per Episode", data=df7, label="DGN")
gd = g.legend(bbox_to_anchor=(1.02, 1), fontsize = 'x-small')
plt.savefig("FortAttackTrainFinal.pdf", bbox_inches='tight')

