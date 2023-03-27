import math as m
import numpy as np
import pandas as pd
import random
from pyqpanda import *

if __name__ == '__main__':
    T = 100  # 时间序列长度
    iterations = 100  # 迭代次数(生成的样本数量)
    Data = []
    for i in range(iterations):
        Seq1 = []
        Seq2 = [0.00001 for _ in range(T)]
        Seqy = []
        rand_index = random.sample(range(0, 100), 2)
        for j in range(T):
            data_1 = random.random()  # 生成0-1的随机数
            Seq1.append(data_1)
        Seq2[rand_index[0]] = 0.99999
        Seq2[rand_index[1]] = 0.99999
        Y_true = Seq1[rand_index[0]] + Seq1[rand_index[1]]
        Seqy.append(Y_true)
        Seq = Seq1+Seq2+Seqy
        Data.append(Seq)
    Data = pd.DataFrame(Data)
    Data.to_csv('lp_100.csv', header=None, index=None)
