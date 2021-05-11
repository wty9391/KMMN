import sys
import os
# sys.path.append('../z_pdf_gaussian_mixture')
import math
import random
import pickle

import numpy as np
import torch
import pandas as pd

from MN import Markov_Network

# ./result/2997

if len(sys.argv) < 3:
    print('Usage: .py result_root_path cuda_index')
    exit(-1)

os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
sample_rate = 0.1
epoch = 9
batch_size = 512
scale = 1
burn_in_epoch = 0
lr_w_p = 1e-0
lr_w_n = 1e-0
lr_c_p = 1e-0
lr_c_n = 1e-0
l2_decay = 5e-5

x_train = pickle.load(open(sys.argv[1] + '/x_train', 'rb'))
b_train_origin = pickle.load(open(sys.argv[1] + '/b_train', 'rb'))
z_train_origin = pickle.load(open(sys.argv[1] + '/z_train', 'rb'))
x_test = pickle.load(open(sys.argv[1] + '/x_test', 'rb'))
z_test_origin = pickle.load(open(sys.argv[1] + '/z_test', 'rb'))

(record_size, x_dimension) = np.shape(x_train)

b_dimension = int(b_train_origin.max()+1)  # include b=0
z_dimension = int(z_train_origin.max()+1)  # include z=0
# print(b_dimension, z_dimension)
# b_train_origin = np.zeros_like(b_train_origin) + b_train_origin.max()

# randomly bid
# np.random.seed(258)
# b_train_origin = np.ceil(np.random.randint(0, b_dimension, size=(record_size, 1)) * \
#                  np.random.randint(0, 101, size=(record_size, 1)) / 100)

MN = Markov_Network(x_dimension=x_dimension, encoder=None, root_path=sys.argv[1],
                    fixed_w_b_z=False, fixed_w_x_omega=False, fixed_c_z=False,
                    z_dimension=z_dimension, b_dimension=b_dimension, device=torch.device("cuda"))
MN.fit(z_train_origin, b_train_origin, x_train,
       sample_rate=sample_rate, epoch=epoch, batch_size=batch_size, scale=scale, burn_in_epoch=burn_in_epoch,
       lr_w_p=lr_w_p, lr_w_n=lr_w_n, lr_c_p=lr_c_p, lr_c_n=lr_c_n, l2_decay=l2_decay)

KL_pdf, WD_pdf, KL_cdf, WD_cdf, ANLP, MSE, omega, c_index = \
    MN.evaluate(x_test.transpose(), z_test_origin,
            path=sys.argv[1]+"/evaluate/{6:.2f}_{0:d}_{1:d}_{7:d}_{8:d}_{9:.6f}_{2:.3f}_{3:.3f}_{4:.3f}_{5:.3f}"
            .format(epoch, batch_size, lr_w_p, lr_w_n, lr_c_p, lr_c_n, sample_rate, scale, burn_in_epoch, l2_decay))

MN.writer.add_scalar('test/pdf/KL', KL_pdf, 0)
MN.writer.add_scalar('test/pdf/WD', WD_pdf, 0)
MN.writer.add_scalar('test/cdf/KL', KL_cdf, 0)
MN.writer.add_scalar('test/cdf/WD', WD_cdf, 0)
MN.writer.add_scalar('test/ANLP', ANLP, 0)
MN.writer.add_scalar('test/MSE', MSE, 0)
MN.writer.add_scalar('test/omega', omega, 0)
MN.writer.add_scalar('test/c_index', c_index, 0)

MN.writer.close()

pickle.dump(MN, open(sys.argv[1]+'/MN', 'wb'))

df = pd.DataFrame(data=[[sys.argv[1].split("/")[-1], 0, 0, 0,
                        'KMMN', KL_pdf, WD_pdf, ANLP, MSE]],
                  columns=['campaign', 'total_record', 'win_record', 'win_rate',
                           'algorithm', 'KL_pdf', 'WD_pdf', 'ANLP', 'MSE'])
output_path = sys.argv[1] + '/../baseline_report.csv'
df.to_csv(output_path, mode='a', index=False, sep='\t', header=not os.path.exists(output_path))











