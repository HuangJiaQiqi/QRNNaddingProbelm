import math as m
import numpy as np
import pandas as pd
import time
from pyqpanda import *
start = time.time()


def numerical_gradient(f, params, x, y):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(params)  # 生成和参数形状相同的全零数组
    for idx in range(params.size):
        tmp_val = params[idx]
        # f(x+h)的计算
        params[idx] = tmp_val + h
        fxh1, ypre = f(params, x, y)
        # f(x-h)的计算
        params[idx] = tmp_val - h
        fxh2, ypre = f(params, x, y)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        params[idx] = tmp_val  # 还原值
    return grad


def MSE(Y, t):
    return 0.5 * (np.sum(Y - t) ** 2)


# 数据输入U_in矩阵
def U_in(qubits, X_t):
    circuit = create_empty_circuit()
    theta_in = m.acos(X_t)
    circuit << RY(qubits[0], theta_in) \
    << RY(qubits[1], theta_in) \
    << RY(qubits[2], theta_in)
    return circuit


# 参数矩阵，有3*6=18个参数
def U_theta(qubits, params):
    circuit = create_empty_circuit()
    for i in range(6):
        circuit << RX(qubits[i], params[3 * i]) \
        << RZ(qubits[i], params[3 * i + 1]) \
        << RX(qubits[i], params[3 * i + 2])
    return circuit


# 哈密顿量模拟第一部分，有6个参数
def H_X(qubits, params):
    circuit = create_empty_circuit()
    for i in range(6):
        circuit << RX(qubits[i], params[i])
    return circuit


# 哈密顿量模拟第二部分，有6个参数
def H_ZZ(qubits, params):
    circuit = create_empty_circuit()
    for i in range(5):
        circuit << CNOT(qubits[i], qubits[i + 1]) \
        << RZ(qubits[i + 1], params[i]) \
        << CNOT(qubits[i], qubits[i + 1])
    circuit << CNOT(qubits[5], qubits[0]) \
    << RZ(qubits[0], params[5]) \
    << CNOT(qubits[5], qubits[0])
    return circuit


# 整个参数线路，共18+6+6=30个参数
def QRNN_VQC(qubits, params):
    params1 = params[0: 18]
    params2 = params[18: 18 + 6]
    params3 = params[18 + 6: 30]
    circuit = create_empty_circuit()
    circuit << U_theta(qubits, params1) \
    << H_X(qubits, params2) \
    << H_ZZ(qubits, params3)
    return circuit


# 损失函数,共30+3个参数，其中前30个为量子线路参数，最后3个为经典参数
def loss(params, X_t, Y_t):
    LOSS = 0
    zhenfu = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    # 数据预处理  注意，这里X_t[n]为预测数据真实值
    X_t_min = min(X_t)
    X_t_max = max(X_t)
    X_t = (X_t - X_t_min) / (X_t_max - X_t_min)
    X_t = X_t.reshape(2, 100)
    for i in range(100):
        qvm = CPUQVM()  # 建立一个局部的量子虚拟机
        qvm.init_qvm()  # 初始化量子虚拟机
        qubits = qvm.qAlloc_many(6)
        prog = QProg()
        circuit = create_empty_circuit()
        X_i = [X_t[0, i],X_t[1, i]]
        # circuit << U_in(qubits, X_t[i])  # 数据输入
        circuit << amplitude_encode([qubits[0], qubits[1], qubits[2]], X_i)
        circuit << amplitude_encode([qubits[3], qubits[4], qubits[5]], zhenfu)
        # print(zhenfu)# 后三个比特的编码
        circuit << QRNN_VQC(qubits, params[0: 30])
        prog << circuit


        qubit0_prob = qvm.prob_run_list(prog, qubits[0], -1)
        qubit1_prob = qvm.prob_run_list(prog, qubits[1], -1)
        qubit2_prob = qvm.prob_run_list(prog, qubits[2], -1)
        # 坍缩到1的概率直接当均值
        qubit0_avrage = qubit0_prob[1]
        # 这里只用第一个比特的概率
        Y_prediction = qubit0_avrage
        # 求后三个比特最后的状态振幅
        zhenfu_2 = qvm.prob_run_list(prog, [qubits[3], qubits[4], qubits[5]], -1)
        zhenfu = np.sqrt(np.array(zhenfu_2))
        # 释放局部虚拟机
        qvm.finalize()

    # LOSS = m.fabs(Y_prediction - X_t[n] )/ X_t[n]
    LOSS = MSE(Y_prediction, Y_t)
    # 数据后处理
    # Y_prediction = Y_prediction * (X_t_max - X_t_min) + X_t_min
    # X_t = (X_t_max - X_t_min) * X_t + X_t_min

    return LOSS, Y_prediction


def Accuarcy(params, zhibiao, n):
    # 测试数据提取
    test_data = pd.read_csv("Test.csv")
    Data = list(test_data.loc[:, zhibiao])
    test_iterations = len(Data) - n - 1  # 测试次数，差分会再少一次
    Data_cha = []

    # 差分预处理
    for i in range(len(Data) - 1):
        Data_cha.append(Data[i + 1] - Data[i])

    #  这里两个用于记录测试集的真实值和预测值
    Y_pre_history_test = []
    Y_true_history_test = []

    # 这个用于存放差值
    Y_pre_test_cha = []

    Ei_2_sum = 0  # 误差平方和初始化
    for j in range(test_iterations):
        zhenfu = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        X_t_cha = np.array(Data_cha[j: j + n + 1])
        X_t = np.array(Data[j: j + n + 2])

        # 数据预处理  注意，这里X_t[n]为预测数据真实值
        X_t_min = min(X_t_cha[0:n])
        X_t_max = max(X_t_cha[0:n])
        X_t_cha = (X_t_cha - X_t_min) / (X_t_max - X_t_min)

        for i in range(n):
            qvm = CPUQVM()  # 建立一个局部的量子虚拟机
            qvm.init_qvm()  # 初始化量子虚拟机
            qubits = qvm.qAlloc_many(6)
            # cbits = qvm.cAlloc_many(6)
            prog = QProg()
            circuit = create_empty_circuit()

            circuit << U_in(qubits, X_t_cha[i])  # 数据输入
            circuit << amplitude_encode([qubits[3], qubits[4], qubits[5]], zhenfu)  # 后三个比特的编码
            circuit << QRNN_VQC(qubits, params[0: 30])
            prog << circuit

            qubit0_prob = qvm.prob_run_list(prog, qubits[0], -1)
            qubit1_prob = qvm.prob_run_list(prog, qubits[1], -1)
            qubit2_prob = qvm.prob_run_list(prog, qubits[2], -1)

            # 坍缩到1的概率直接当均值
            qubit0_avrage = qubit0_prob[1]

            # 这里只用第一个比特的概率
            Y_prediction = qubit0_avrage

            # 求后三个比特最后的状态振幅，这里还需要修改，使用的模方再开根，不含复数
            zhenfu_2 = qvm.prob_run_list(prog, [qubits[3], qubits[4], qubits[5]], -1)
            zhenfu = np.sqrt(np.array(zhenfu_2))

            # 释放局部虚拟机
            qvm.finalize()

        # 数据后处理
        Y_prediction = Y_prediction * (X_t_max - X_t_min) + X_t_min
        X_t_cha = (X_t_max - X_t_min) * X_t_cha + X_t_min

        # 这里得到的Y_prediciton为差值，需要进行处理
        Y_prediction = X_t[n] + Y_prediction

        # Y_pre_history_test.append(Y_prediction * sum + Y_tmin)
        Y_pre_history_test.append(Y_prediction)
        Y_true_history_test.append(X_t[n + 1])

        # Ei = m.fabs(Y_t[n - 1] - Y_prediction * sum) / Y_t[n - 1]  # 计算误差
        if X_t[n] == 0:
            Ei = 0
        else:
            Ei = m.fabs(X_t[n] - Y_prediction) / X_t[n]  # 计算误差
        Ei_2 = Ei * Ei
        Ei_2_sum = Ei_2_sum + Ei_2

    accuarcy = 1 - m.sqrt(Ei_2_sum / test_iterations)
    return accuarcy, Y_pre_history_test, Y_true_history_test


if __name__ == "__main__":
    params = np.random.randn(30)
    params = list(params)  # 转换成列表用于添加元素
    params = np.array(params)  # 再转换为数组
    df = pd.read_csv("lp_100.csv", header=None)  # 读取长程依赖问题数据集
    iterations = 100
    for i in range(iterations):
        data = df.loc[i]
        Seq = np.array(data.iloc[0:200])
        # Seq = Seq.reshape(2, 100)
        Y_true = data.iloc[-1]
        # 求梯度
        grad = numerical_gradient(loss, params, Seq, Y_true)
        # 求损失函数值（用于存储），此时预测值Y_pre_n均为差值
        LOSS, Y_pre_cha = loss(params, Seq, Y_true)
        # 当天预测值 = 当天前一天的真实值 + 差值
        Y_pre = X_t_in[n] + Y_pre_cha

        # 记录损失函数
        loss_history.append(LOSS)
        # 记录预测值
        Y_pre_history.append(Y_pre)
        # 梯度下降更新参数
        if i < 301:
            params = params - lr * grad
        # 记录参数
        params_history.append(list(params))
    params_P = params
