import sys
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack

from util import encoder
import util.truthful_bidder as truthful_bidder

# /home/wty/datasets/make-ipinyou-data/2997/train.log.txt /home/wty/datasets/make-ipinyou-data/2997/test.log.txt /home/wty/datasets/make-ipinyou-data/2997/featindex.txt ./result/2997


if len(sys.argv) < 5:
    print('Usage: .py trian_log_path test_log_path feat_path result_root_path')
    exit(-1)

read_batch_size = 1e6

f_train_log = open(sys.argv[1], 'r', encoding="utf-8")
f_test_log = open(sys.argv[2], 'r', encoding="utf-8")

f_train_yzbx = open(sys.argv[4] + '/train.yzbx.txt', 'w+', encoding="utf-8")  # for DLF
f_test_yzbx = open(sys.argv[4] + '/test.yzbx.txt', 'w+', encoding="utf-8")  # for DLF

# init name_col
name_col = {}
s = f_train_log.readline().split('\t')
for i in range(0, len(s)):
    name_col[s[i].strip()] = i

ipinyou = encoder.Encoder_ipinyou(sys.argv[3], name_col)
X_train_raw = []
X_train = csr_matrix((0, len(ipinyou.feat)), dtype=np.int8)
Y_train = np.zeros((0, 1), dtype=np.int8)
B_train = np.zeros((0, 1), dtype=np.int16)
Z_train = np.zeros((0, 1), dtype=np.int16)
X_test_raw = []
X_test = csr_matrix((0, len(ipinyou.feat)), dtype=np.int8)
Y_test = np.zeros((0, 1), dtype=np.int8)
B_test = np.zeros((0, 1), dtype=np.int16)
Z_test = np.zeros((0, 1), dtype=np.int16)

count = 0
f_train_log.seek(0)
f_train_log.readline()  # first line is header
for line in f_train_log:
    X_train_raw.append(line)
    count += 1
    if count % read_batch_size == 0:
        X_train = vstack((X_train, ipinyou.encode(X_train_raw)))
        Y_train = np.vstack((Y_train, ipinyou.get_col(X_train_raw, "click")))
        Z_train = np.vstack((Z_train, ipinyou.get_col(X_train_raw, "payprice")))
        X_train_raw = []
if X_train_raw:
    X_train = vstack((X_train, ipinyou.encode(X_train_raw)))
    Y_train = np.vstack((Y_train, ipinyou.get_col(X_train_raw, "click")))
    Z_train = np.vstack((Z_train, ipinyou.get_col(X_train_raw, "payprice")))
    X_train_raw = []

count = 0
f_test_log.seek(0)
f_test_log.readline()  # first line is header
for line in f_test_log:
    X_test_raw.append(line)
    count += 1
    if count % read_batch_size == 0:
        X_test = vstack((X_test, ipinyou.encode(X_test_raw)))
        Y_test = np.vstack((Y_test, ipinyou.get_col(X_test_raw, "click")))
        Z_test = np.vstack((Z_test, ipinyou.get_col(X_test_raw, "payprice")))
        X_test_raw = []
if X_test_raw:
    X_test = vstack((X_test, ipinyou.encode(X_test_raw)))
    Y_test = np.vstack((Y_test, ipinyou.get_col(X_test_raw, "click")))
    Z_test = np.vstack((Z_test, ipinyou.get_col(X_test_raw, "payprice")))
    X_test_raw = []

(train_size, x_dimension) = np.shape(X_train)
(test_size, _) = np.shape(X_test)

z_dimension = int(max(Z_train.max(), Z_test.max())+1)
b_dimension = z_dimension + 1
# z_dimension = 301
# b_dimension = 302
b_max = b_dimension - 1
# win_rate = 0.5


np.random.seed(258)

# randomly generate win-lose split for training set
# for winning cases, randomly generate b = z + alpha*(b_max-z) where alpha is (0,1)
# for losing cases, randomly generate  b = z - alpha*(z-z_min) where alpha is (0,1)
# B_train = np.zeros((train_size, 1), dtype=np.int16)
# is_win = np.random.choice([False, True], train_size, p=[1 - win_rate, win_rate])
# is_lose = is_win == False
# is_win_index = is_win.reshape(-1, )
# is_lose_index = is_lose.reshape(-1, )
# win_size = is_win.astype(np.int8).sum()
# lose_size = is_lose.astype(np.int8).sum()
# assert win_size + lose_size == train_size
#
# b_max = np.ones((win_size, 1), dtype=np.int16) * (b_dimension - 1)
# z_min = np.ones((lose_size, 1), dtype=np.int16) * 0
# alpha = np.random.random(size=(train_size, 1))
#
#
# B_train[is_win_index, :] = np.ceil(Z_train[is_win_index, :] + alpha[is_win_index, :] *
#                                    (b_max - Z_train[is_win_index, :]))
# B_train[is_lose_index, :] = np.floor(Z_train[is_lose_index, :] - alpha[is_lose_index, :] *
#                                      (Z_train[is_lose_index, :] - z_min))


# threshold_b = 100
# bidder = truthful_bidder.Truthful_bidder(max_iter=10)
# _, alpha = bidder.fit(X_train, Y_train, Z_train)
# bidder.evaluate(X_test, Y_test)
# B_train = np.asarray(bidder.bid(X_train)).reshape((-1, 1))
# rerandom_index = B_train >= threshold_b
# B_train[rerandom_index] = np.ceil(np.random.randint(threshold_b, b_dimension, size=(rerandom_index.sum(), 1))).reshape((-1,))
# B_train[B_train > b_max] = b_max
# B_test = np.asarray(bidder.bid(X_test)).reshape((-1, 1))
# B_test[B_test > b_max] = b_max
np.random.seed(258)
bid_rate = 0.25
# B_train = np.ceil(np.random.randint(1, b_dimension, size=(train_size, 1)))
B_train = np.zeros((train_size, 1), dtype=np.int16)
B_train = np.ceil(np.random.random(size=(train_size, 1)) * Z_train)
is_bid = np.random.choice([False, True], train_size, p=[1 - bid_rate, bid_rate])
B_train[is_bid] = np.ceil(np.random.randint(1, b_dimension, size=(is_bid.sum(), 1)))

# bid price for test set is useless, thus randomly generate
B_test = np.ceil(np.random.randint(1, b_dimension, size=(test_size, 1)))

# B_train = np.ceil(np.random.randint(1, b_dimension, size=(train_size, 1)) * \
#                   np.random.randint(1, 101, size=(train_size, 1)) / 100)
# bid price for test set is useless, thus randomly generate
# B_test = np.ceil(np.random.randint(1, b_dimension, size=(test_size, 1)) * \
#                  np.random.randint(1, 101, size=(test_size, 1)) / 100)

for i in range(train_size):
    f_train_yzbx.write(str(Y_train[i, 0]) + " ")
    f_train_yzbx.write(str(Z_train[i, 0]) + " ")
    f_train_yzbx.write(str(int(B_train[i, 0])) + " ")
    f_train_yzbx.write(":1 ".join(
        "{0:d}".format(n) for n in X_train.getrow(i).indices.tolist()[0:16]))  # The length of x in DLF is fixed
    f_train_yzbx.write(":1\n")

for i in range(test_size):
    f_test_yzbx.write(str(Y_test[i, 0]) + " ")
    f_test_yzbx.write(str(Z_test[i, 0]) + " ")
    f_test_yzbx.write(str(int(B_test[i, 0])) + " ")
    f_test_yzbx.write(":1 ".join(
        "{0:d}".format(n) for n in X_test.getrow(i).indices.tolist()[0:16]))  # The length of x in DLF is fixed
    f_test_yzbx.write(":1\n")

pickle.dump(X_train, open(sys.argv[4] + '/x_train', 'wb'))
pickle.dump(Y_train, open(sys.argv[4]+'/y_train', 'wb'))
pickle.dump(B_train, open(sys.argv[4] + '/b_train', 'wb'))
pickle.dump(Z_train, open(sys.argv[4] + '/z_train', 'wb'))
pickle.dump(X_test, open(sys.argv[4] + '/x_test', 'wb'))
pickle.dump(Y_test, open(sys.argv[4]+'/y_test', 'wb'))
pickle.dump(Z_test, open(sys.argv[4] + '/z_test', 'wb'))

f_train_log.close()
f_test_log.close()
f_train_yzbx.close()
f_test_yzbx.close()
