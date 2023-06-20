import datetime
import pandas as pd
import pulp as pl
import multiprocessing as mp
from multiprocessing import freeze_support
freeze_support()

stime = datetime.datetime.now()
# 读入数据
df1 = pd.read_excel(r"C:\Users\HCRD\Desktop\sheet1.xlsx", sheet_name="Sheet1")
df2 = pd.read_excel(r"C:\Users\HCRD\Desktop\sheet2.xlsx", sheet_name="Sheet1")
df = pd.read_excel(r"C:\Users\HCRD\Desktop\sheet3.xlsx", sheet_name="Sheet1")
I = df1["po_id"]  # 任务的集合
J = df2["stat_id"]  # 站点的集合
# 人数的记录
h_dict = {}
for i in range(len(df2)):
    stat_id = df2.loc[i, "stat_id"]
    h = df2.loc[i, "h"]
    h_dict[stat_id] = h
# 实绩的记录
P_dict = {}
for i in range(len(df)):
    po_id = df.loc[i, "po_id"]
    stat_id = df.loc[i, "stat_id"]
    P_ij = df.loc[i, "P_ij"]
    if po_id not in P_dict:
        P_dict[po_id] = {}
    P_dict[po_id][stat_id] = P_ij
# 距离的记录
d_dict = {}
for i in range(len(df)):
    po_id = df.loc[i, "po_id"]
    stat_id = df.loc[i, "stat_id"]
    d = df.loc[i, "d"]
    if po_id not in d_dict:
        d_dict[po_id] = {}
    d_dict[po_id][stat_id] = d
M = 1e6  # 大M法中的大M值

# 定义子进程函数
def solve_subset(subset):
    print(f"Starting subset {subset}")
    # 创建子模型
    m_subset = pl.LpProblem(f"Subset {subset}", pl.LpMinimize)

    # 添加子模型的决策变量，与主模型相同
    x_subset = pl.LpVariable.dicts("x", [(i, j) for i in subset for j in J], cat='Binary')
    q_bar_subset = pl.LpVariable.dicts("q_bar", J, lowBound=0)
    q_max_subset = pl.LpVariable("q_max", lowBound=0)
    q_min_subset = pl.LpVariable("q_min", lowBound=0)
    u_subset = pl.LpVariable.dicts("u", J, cat='Binary')
    v_subset = pl.LpVariable.dicts("v", J, cat='Binary')

    # 添加子模型的目标函数，与主模型相同
    R_subset = q_max_subset - q_min_subset
    D_subset = pl.lpSum(d_dict[i][j] * x_subset[(i, j)] for i in subset for j in J)
    m_subset += R_subset

    # 添加子模型的限制条件，与主模型相同
    for i in subset:
        m_subset += pl.lpSum(x_subset[(i, j)] for j in J) == 1

    for i in subset:
        for j in J:
            if d_dict[i][j] > sum(d_dict[i].values()) / len(J):
                m_subset += x_subset[(i, j)] == 0

    for j in J:
         m_subset += q_bar_subset[j] == pl.lpSum(d_dict[i][j] * x_subset[(i, j)] for i in subset) / int(h_dict[j])

    for j in J:
        m_subset += q_bar_subset[j] <= q_max_subset
        m_subset += q_bar_subset[j] + M * (1 - u_subset[j]) >= q_max_subset
    m_subset += pl.lpSum(u_subset[j] for j in J) >= 1

    for j in J:
        m_subset += q_bar_subset[j] >= q_min_subset
        m_subset += q_bar_subset[j] - M * (1 - v_subset[j]) <= q_min_subset
    m_subset += pl.lpSum(v_subset[j] for j in J) >= 1

    # 求解子模型
    m_subset.solve()

    # 返回子模型的结果
    return {i: j for i in subset for j in J if x_subset[(i, j)].value() > 0.5}

# 创建模型
m = pl.LpProblem('Multi-objective Optimization Model', pl.LpMinimize)
# 创建决策变量
x = pl.LpVariable.dicts("x", [(i, j) for i in I for j in J], cat='Binary')
q_bar = pl.LpVariable.dicts("q_bar", J, lowBound=0)
q_max = pl.LpVariable("q_max", lowBound=0)
q_min = pl.LpVariable("q_min", lowBound=0)
u = pl.LpVariable.dicts("u", J, cat='Binary')
v = pl.LpVariable.dicts("v", J, cat='Binary')

# 定义目标函数
R = q_max - q_min
D = pl.lpSum(d_dict[i][j] * x[(i, j)] for i in I for j in J)
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
    m += q_bar[j] == pl.lpSum(d_dict[i][j] * x[(i, j)] for i in I) / int(h_dict[j])
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
# 将任务集合I划分为多个子集
I_subsets = []
n_subsets = mp.cpu_count()  # 子集的数量等于CPU核心数
n_per_subset = len(I) // n_subsets
for i in range(n_subsets):
    if i == n_subsets - 1:  # 最后一个子集包含所有剩余的任务
        I_subsets.append(I[i * n_per_subset:])
    else:
        I_subsets.append(I[i * n_per_subset:(i + 1) * n_per_subset])

# 创建进程池
pool = mp.Pool(processes=n_subsets)
# 在进程池中并行运行子进程函数，并获得结果
results = pool.map(solve_subset, I_subsets)
# 关闭进程池
pool.close()
pool.join()
# 将子模型的结果合并
x_values = {}
for result in results:
    x_values.update(result)
# 将子模型的结果添加到主模型中
for (i, j), value in x_values.items():
    x[(i, j)].value = value
# 再次加入R的条件，以距离最小来求解
R_min = pl.value(R)
m += q_max - q_min == R_min
m.setObjective(D)
# 求解模型
m.solve()
# 输出结果
if m.status == 1:
    print("Optimal value of R:", pl.value(R))
    print("Optimal value of D:", pl.value(D))
    for i in I:
        for j in J:
            if x[(i, j)].value() > 0.5:
                print(f"Task {i} is assigned to site {j}")
# 统计时间
etime = datetime.datetime.now()
print(etime - stime)
