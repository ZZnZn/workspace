import datetime
import pandas as pd
import pulp as pl

# 读入数据
def read_data():
    df1 = pd.read_excel(".\\spot_id.xlsx", sheet_name="Sheet1")
    df2 = pd.read_excel(".\\stat.xlsx", sheet_name="Sheet1")
    df = pd.read_excel(".\\result.xlsx", sheet_name="Sheet1")
    return df1, df2, df
# 人数的记录
def get_h_dict(df2):
    h_dict = {}
    for i in range(len(df2)):
        stat_id = df2.loc[i, "stat_id"]
        h = df2.loc[i, "h"]
        h_dict[stat_id] = h
    return h_dict
# 实绩的记录
def get_P_dict(df):
    P_dict = {}
    for i in range(len(df)):
        spot_id = df.loc[i, "spot_id"]
        stat_id = df.loc[i, "stat_id"]
        P_ij = df.loc[i, "P_ij"]
        if spot_id not in P_dict:
            P_dict[spot_id] = {}
        P_dict[spot_id][stat_id] = P_ij
    return P_dict
# 距离的记录
def get_d_dict(df):
    d_dict = {}
    for i in range(len(df)):
        spot_id = df.loc[i, "spot_id"]
        stat_id = df.loc[i, "stat_id"]
        d = df.loc[i, "d"]
        if spot_id not in d_dict:
            d_dict[spot_id] = {}
        d_dict[spot_id][stat_id] = d
    return d_dict

import csv
def solve_model_fixedH(I, J, H, P_dict, d_dict):
    # 大M法中的大M值
    M = 1e6
    # 创建模型
    m = pl.LpProblem('Multi-objective Optimization Model', pl.LpMinimize)
    # 创建决策变量
    x = pl.LpVariable.dicts("x", [(i, j) for i in I for j in J], cat='Binary')
    h = pl.LpVariable.dicts("h", J, cat='Integer')
    q_bar = pl.LpVariable.dicts("q_bar", J, lowBound=0, cat='Integer')
    q_max = pl.LpVariable("q_max", lowBound=0, cat='Integer')
    q_min = pl.LpVariable("q_min", lowBound=0, cat='Integer')
    u = pl.LpVariable.dicts("u", J, cat='Binary')
    v = pl.LpVariable.dicts("v", J, cat='Binary')
    # 定义目标函数
    R = q_max - q_min
    m += R
    # 加入限制条件
    # 任务分配
    for i in I:
        m += pl.lpSum(x[(i, j)] for j in J) == 1
    # 只取距离小于等于平均的站
    for i in I:
        for j in J:
            if d_dict[i][j] > sum(d_dict[i].values()) / len(J):
                m += x[(i, j)] == 0
    # 平均人效计算
    for j in J:
        m += q_bar[j] * h[j] == pl.lpSum(P_dict[i][j] * x[(i, j)] for i in I)
        m += q_bar[j] * h[j] + 1  <= pl.lpSum(P_dict[i][j] * x[(i, j)] for i in I)
    # 平均人效最大值计算
    for j in J:
        m += q_bar[j] <= q_max
        m += q_bar[j] + M * (1 - u[j]) >= q_max
    m += pl.lpSum(u[j] for j in J) >= 1
    # 平均人效最小值计算
    for j in J:
        m += q_bar[j] >= q_min
        m += q_bar[j] - M * (1 - v[j]) <= q_min
    m += pl.lpSum(v[j] for j in J) >= 1
    # 人数限制
    m += pl.lpSum(h[j] for j in J) == H
    # 求解模型
    m.solve()
    # 再次加入R最小的条件，以距离最小来求解
    R_min = pl.value(R)
    m += q_max - q_min == R_min
    m += pl.lpSum(d_dict[i][j] * x[(i, j)] for i in I for j in J)
    m.setObjective(pl.lpSum(d_dict[i][j] * x[(i, j)] for i in I for j in J))
    m.solve()
    # 汇报结果
    value_dict = m.variablesDict()
    print("Optimal value of R:",
          pl.value(value_dict["q_max"]) - pl.value(value_dict["q_min"]))
    print("Optimal value of D:", pl.value(m.objective))

    with open('spot_id_modified.csv','w',encoding='UTF8',newline='') as f:
        writer = csv.writer(f)
        data = ['spot_id','stat_id']
        writer.writerow(data)
        for idx,i in enumerate(I):
            for j in J:
                x_string = "x_(" + str(i) + ",_" + str(j) + ")"
                if pl.value(value_dict[x_string]) > 0.5:
                    writer.writerow([i,j])
            if idx % 100 == 0:
                print(idx)

def OptimizationModel():
    stime = datetime.datetime.now()
    # 读入数据
    H = 1173
    df1, df2, df = read_data()
    I = df1["spot_id"]  # 任务的集合
    J = df2["stat_id"]  # 站点的集合
    P_dict = get_P_dict(df)
    d_dict = get_d_dict(df)
    solve_model_fixedH(I, J, H, P_dict, d_dict)

    # 统计时间
    etime = datetime.datetime.now()
    print(etime - stime)

OptimizationModel()
