import numpy as np

# 储存概率
possibility = np.array([0.0 for _ in range(301)])
possibility_pre = np.array([0.0 for _ in range(301)])

# 定义转移矩阵
transition_matrix = np.zeros((102,102))
for i in range(0,102):
    if i in range(0,50):
        transition_matrix[i][i+1] = 0.98
        transition_matrix[i][100] = 0.02 * 0.65
        transition_matrix[i][101] = 0.02 * 0.35
    elif i in range(50,100):
        transition_matrix[i][i+1] = 1 - 0.02 * (i - 48)
        transition_matrix[i][100] = 0.02 * (i - 48) * 0.65
        transition_matrix[i][101] = 0.02 * (i - 48) * 0.35
    elif i == 100 :
        transition_matrix[i][1] = 0.98
        transition_matrix[i][100] = 0.02 * 0.65
        transition_matrix[i][101] = 0.02 * 0.35

# 定义初始概率矩阵
initial_condition = np.zeros((1,102))
initial_condition[0][0] = 1

# 矩阵乘法
for i in range(1,300):
    initial_condition = np.matmul(initial_condition, transition_matrix)
    possibility[i] = initial_condition[0][101]
    possibility_pre[i] = sum(possibility)

possibility[300] = 1 - np.sum(possibility)
possibility_pre[300] = sum(possibility)

# 计算期望
expectation = sum([i*possibility[i] for i in range(1,301)])
print(expectation)
