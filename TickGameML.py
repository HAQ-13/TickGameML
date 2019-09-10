from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd

fn1 = 'data/y_tb_label.csv'
fn2 = 'data/y_scaling_law_features.csv'
df1 = pd.read_csv(fn1)
df2 = pd.read_csv(fn2)

# 目标结果一一对应数据
df = df2.merge(df1, on='trading_time')
df['triple_barrier_label'].replace({'win': 10, 'loss': -10, 'stop holding': 0}, inplace=True)

# 整体数据
data_X, targets_Y = (df[['N_os/N_dc', 'Delta_t_os/Delta_t_dc', 'Sigma_V_os/Sigma_V_dc']], df['triple_barrier_label'])
# 前9/10的数据作为train数据
data_train, targets_train = (data_X[:(len(df) // 10) * 9], targets_Y[:(len(df) // 10) * 9])
# 标准化数据
scaler = preprocessing.StandardScaler().fit(data_train)
data_train_scaled = scaler.transform(data_train)
# 后1/10的数据作为test数据
data_test, targets_test = (data_X[(len(df) // 10) * 9:], targets_Y[(len(df) // 10) * 9:])
data_test_scaled = scaler.transform(data_test)
data_X_scaled = scaler.transform(data_X)

# SVC
clf_svc = SVC(gamma='auto')
clf_svc.fit(data_train_scaled, targets_train)
print('SVC score of test data:', clf_svc.score(data_test_scaled, targets_test))
print('SVC score of all data:', clf_svc.score(data_X_scaled, targets_Y))

# RFC
clf_rfc = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf_rfc.fit(data_train_scaled, targets_train)
print('RFC score of test data:', clf_rfc.score(data_test_scaled, targets_test))
print('RFC score of all data:', clf_rfc.score(data_X_scaled, targets_Y))
