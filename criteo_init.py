import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

import util.truthful_bidder as truthful_bidder

def to_dict_values(df_view, features):
    return [dict([(_[0] + str(_[1]), 1) for _ in zip(features, l)]) for l in df_view.values]

# /home/wty/datasets/make-criteo-data/criteo_attribution_dataset.tsv.gz ./result/criteo

if len(sys.argv) < 3:
    print('Usage: .py log_path result_root_path')
    exit(-1)

f_train_yzbx = open(sys.argv[2] + '/train.yzbx.txt', 'w+', encoding="utf-8")  # for DLF
f_test_yzbx = open(sys.argv[2] + '/test.yzbx.txt', 'w+', encoding="utf-8")  # for DLF
f_featindex = open(sys.argv[2] + '/featindex.txt', 'w+', encoding="utf-8")  # for DLF

print("Start to load log")
df = pd.read_csv(sys.argv[1], sep='\t', compression='gzip')
df['day'] = np.floor(df.timestamp / 86400.).astype(int)
FEATURES = ['campaign', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8']
z_index = 'cost'
z_scale = 1e5
y_index = 'click'
z_max = np.percentile(df[z_index].values.reshape((-1, 1)) * z_scale, 99)
z_dimension = np.ceil(z_max) + 1
b_dimension = z_dimension + 1
train_test_split_day = 24

hasher = FeatureHasher(n_features=2**16, alternate_sign=False)

df[z_index] = np.ceil(df[z_index].values * z_scale)
df[z_index] = np.clip(df[z_index].values, 0, z_max)
df_train = df[df.day < train_test_split_day]
df_test = df[df.day >= train_test_split_day]
train_size, _ = df_train.shape
test_size, _ = df_test.shape

print("Start to perform feature hash")
x_train = hasher.fit_transform(to_dict_values(df_train[FEATURES], FEATURES)).tocsr()
x_test = hasher.transform(to_dict_values(df_test[FEATURES], FEATURES)).tocsr()
z_train = np.ceil(df_train[z_index].values.reshape((-1, 1))).astype(np.int16)
z_test = np.ceil(df_test[z_index].values.reshape((-1, 1))).astype(np.int16)
y_train = np.ceil(df_train[y_index].values.reshape((-1, 1))).astype(np.int8)
y_test = np.ceil(df_test[y_index].values.reshape((-1, 1))).astype(np.int8)

bidder = truthful_bidder.Truthful_bidder(max_iter=50)
_, alpha = bidder.fit(x_train, y_train, z_train)
bidder.evaluate(x_train, y_train)
bidder.evaluate(x_test, y_test)

np.random.seed(258)
bid_rate = 0.25
# B_train = np.ceil(np.random.randint(1, b_dimension, size=(train_size, 1)))
b_train = np.zeros((train_size, 1), dtype=np.int16)
b_train = np.ceil(np.random.random(size=(train_size, 1)) * z_train)
is_bid = np.random.choice([False, True], train_size, p=[1 - bid_rate, bid_rate])
b_train[is_bid] = np.ceil(np.random.randint(1, b_dimension, size=(is_bid.sum(), 1)))

# bid price for test set is useless, thus randomly generate
b_test = np.ceil(np.random.randint(1, b_dimension, size=(test_size, 1)))

print("Start save yzbx.txt")
# for DLF
for i in range(train_size):
    f_train_yzbx.write(str(y_train[i, 0]) + " ")
    f_train_yzbx.write(str(z_train[i, 0]) + " ")
    f_train_yzbx.write(str(int(b_train[i, 0])) + " ")
    f_train_yzbx.write(":1 ".join(
        "{0:d}".format(n) for n in x_train.getrow(i).indices.tolist()[0:16]))  # The length of x in DLF is fixed
    f_train_yzbx.write(":1\n")

for i in range(test_size):
    f_test_yzbx.write(str(y_test[i, 0]) + " ")
    f_test_yzbx.write(str(z_test[i, 0]) + " ")
    f_test_yzbx.write(str(int(b_test[i, 0])) + " ")
    f_test_yzbx.write(":1 ".join(
        "{0:d}".format(n) for n in x_test.getrow(i).indices.tolist()[0:16]))  # The length of x in DLF is fixed
    f_test_yzbx.write(":1\n")

print("Start save featindex.txt")
f_featindex.write("truncate\t0\n")
feat_index = {f+1: {} for f in range(len(FEATURES))}  # feature : {index : value}
count = 1
for i in range(train_size):
    x = x_train.getrow(i).indices.tolist()
    # assert len(x) == len(FEATURES)
    for j in range(len(x)):
        feature = j + 1
        index = x[j]
        if index not in feat_index[feature]:
            feat_index[feature][index] = count
            count += 1

for feature, index_value in feat_index.items():
    for index, value in index_value.items():
        f_featindex.write("{0:d}:{1:d}\t{2:d}\n".format(feature, value, index))

print("Start save train/test.log.txt")
# for KMDT
FEAT_NAME = []
FEAT_NAME.extend(FEATURES)
FEAT_NAME.append(z_index)
df_train[FEAT_NAME].to_csv(sys.argv[2]+'/train.log.txt', index=False, sep='\t', header=True)
df_test[FEAT_NAME].to_csv(sys.argv[2]+'/test.log.txt', index=False, sep='\t', header=True)

print("Start to save processed data")
pickle.dump(x_train, open(sys.argv[2]+'/x_train', 'wb'))
pickle.dump(y_train, open(sys.argv[2]+'/y_train', 'wb'))
pickle.dump(b_train, open(sys.argv[2]+'/b_train', 'wb'))
pickle.dump(z_train, open(sys.argv[2]+'/z_train', 'wb'))
pickle.dump(x_test, open(sys.argv[2]+'/x_test', 'wb'))
pickle.dump(y_test, open(sys.argv[2]+'/y_test', 'wb'))
pickle.dump(z_test, open(sys.argv[2]+'/z_test', 'wb'))

f_train_yzbx.close()
f_test_yzbx.close()
f_featindex.close()

