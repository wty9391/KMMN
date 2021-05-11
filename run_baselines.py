import sys
import os

# sys.path.append('../z_pdf_gaussian_mixture')
import math
import random
import pickle
from prettytable import PrettyTable
import numpy as np
# import torch
# import seaborn as sns
import matplotlib.pyplot as plt
# from lifelines import KaplanMeierFitter
import pandas as pd

from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_squared_error

from baselines import KM, KMDT, CLR
from util import util
import util.truthful_bidder as truthful_bidder

from baselines.CGMM.softmax_censored_gaussian_mixture import SoftmaxCensoredGaussianMixture
import baselines.CGMM.init_util as init_util
from baselines.CGMM.dataset_processor import Censored_processor

# ./result/2997 /home/wty/datasets/make-ipinyou-data/
# ./result/criteo ./result/

if len(sys.argv) < 3:
    print('Usage: .py result_root_path ')
    exit(-1)

OFROOT = '../result/SurvivalModel/'
epoch = 10

x_train = pickle.load(open(sys.argv[1] + '/x_train', 'rb'))
y_train = pickle.load(open(sys.argv[1] + '/y_train', 'rb'))
b_train_origin = pickle.load(open(sys.argv[1] + '/b_train', 'rb'))
z_train_origin = pickle.load(open(sys.argv[1] + '/z_train', 'rb'))
x_test = pickle.load(open(sys.argv[1] + '/x_test', 'rb'))
y_test = pickle.load(open(sys.argv[1] + '/y_test', 'rb'))
z_test_origin = pickle.load(open(sys.argv[1] + '/z_test', 'rb'))

(record_size, x_dimension) = np.shape(x_train)
(test_size, _) = np.shape(x_test)

b_dimension = int(b_train_origin.max() + 1)  # include b=0
z_dimension = int(z_train_origin.max() + 1)  # include z=0
# b_dimension = 301  # include b=0
# z_dimension = 301  # include z=0
campaign = sys.argv[1].split("/")[-1]

win = b_train_origin > z_train_origin
win_rate = win.sum() / record_size
print("winning rate {0:.2f}%".format(win_rate * 100))

report = PrettyTable(['campaign', 'total_record', 'win_record', 'win_rate',
                      'algorithm', 'KL_pdf', 'WD_pdf', 'ANLP', 'MSE'])
zs = list(range(z_dimension))
# calculate truth_pdf
truth_pdf = []
(unique_z, counts_z) = np.unique(z_test_origin, return_counts=True)  # the unique has been sorted
unique_z = unique_z.tolist()

for i in range(z_dimension):
    count = counts_z[unique_z.index(i)] if i in unique_z else 0  # in case of dividing 0
    truth_pdf.append(count / test_size)

# KM
print("==========start to train KM==========")
km = KM.KM(max_market_price=z_dimension)
km.fit(z_train_origin, b_train_origin)

kl_pdf_km = entropy(truth_pdf, km.pdf)
wd_pdf_km = wasserstein_distance(truth_pdf, km.pdf)
anlp_km = km.anlp(x_test, z_test_origin)
mse_km = mean_squared_error(z_test_origin, km.predict_z(x_test))
report.add_row([campaign, record_size, win.sum(), win_rate,
                "KM", kl_pdf_km, wd_pdf_km, anlp_km, mse_km])

# KMDT
print("==========start to train KMDT==========")
kmdt = KMDT.KMDT(IFROOT=sys.argv[2], result_root=sys.argv[1], OFROOT=OFROOT, max_market_price=z_dimension)
kmdt.train()
mse_kmdt, anlp_kmdt, pdf_kmdt = kmdt.evaluate(x_test, z_test_origin)
kl_pdf_kmdt = entropy(truth_pdf, pdf_kmdt)
wd_pdf_kmdt = wasserstein_distance(truth_pdf, pdf_kmdt)
report.add_row([campaign, record_size, win.sum(), win_rate,
                "KMDT", kl_pdf_kmdt, wd_pdf_kmdt, anlp_kmdt, mse_kmdt])

# CLR
print("==========start to train CLR==========")
clr = CLR.MixtureModel(x_dimension, b_train_origin, variance=5)
clr.fit(z_train_origin, x_train, epoch=epoch, batch_size=1024, eta_w=5e-4, verbose=1)

clr_predict_z = np.round(clr.predict(x_test)[:, 0]).tolist()
clr_result = Censored_processor._count(clr_predict_z, min_price=0, max_price=z_dimension)
pdf_clr = [clr_result["pdf"][z] if clr_result["pdf"][z] != 0 else 1e-6 for z in zs]
cdf_clr = [clr_result["cdf"][z] if clr_result["cdf"][z] != 0 else 1e-6 for z in zs]
kl_pdf_clr = entropy(truth_pdf, pdf_clr)
wd_pdf_clr = wasserstein_distance(truth_pdf, pdf_clr)
mse_clr = mean_squared_error(z_test_origin, clr_predict_z)
anlp_clr = clr.anlp(x_test, z_test_origin)
report.add_row([campaign, record_size, win.sum(), win_rate,
                "CLR", kl_pdf_clr, wd_pdf_clr, anlp_clr, mse_clr])

# CGMM
print("==========start to train CGMM==========")
_, mean, variance = init_util.init_gaussian_mixture_parameter(5, min_z=1, max_z=250)
cgmm = SoftmaxCensoredGaussianMixture(b_train_origin, x_dimension, len(mean),
                                      min_z=0, max_z=z_dimension, mean=mean, variance=variance)
cgmm.fit(z_train_origin, x_train, z_test_origin, x_test,
         sample_rate=0.5, epoch=epoch, batch_size=1024, eta_w=1e-1, eta_mean=1e1, eta_variance=1e2, labda=0.0, verbose=0)
print(cgmm)

pdf_cgmm = cgmm.pdf_overall(zs, x_test)
cdf_cgmm = cgmm.cdf_overall(zs, x_test)
kl_pdf_cgmm = entropy(truth_pdf, pdf_cgmm)
wd_pdf_cgmm = wasserstein_distance(truth_pdf, pdf_cgmm)
mse_cgmm = mean_squared_error(z_test_origin, cgmm.predict_z(x_test))
anlp_cgmm = cgmm.ANLP(z_test_origin, x_test)
report.add_row([campaign, record_size, win.sum(), win_rate,
                "CGMM", kl_pdf_cgmm, wd_pdf_cgmm, anlp_cgmm, mse_cgmm])

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(zs, km.pdf, color='tab:blue', label='KM')
ax1.plot(zs, truth_pdf, color='tab:purple', label='truth')
ax1.plot(zs, pdf_kmdt, color='tab:red', label='KMDT')
ax1.plot(zs, pdf_cgmm, color='tab:orange', label='CGMM')
ax1.plot(zs, pdf_clr, color='tab:green', label='CLR')
ax1.set_ylabel("pdf")
# ax1.set_ylim(0, 0.15)
ax1.legend()
# f.set_size_inches(7, 7 / 16 * 9)

ax2.plot(zs, km.w, color='tab:blue', label='KM')
ax2.plot(zs, util.Util.pdf_to_cdf(truth_pdf), color='tab:purple', label='truth')
ax2.plot(zs, util.Util.pdf_to_cdf(pdf_kmdt), color='tab:red', label='KMDT')
ax2.plot(zs, util.Util.pdf_to_cdf(pdf_cgmm), color='tab:orange', label='CGMM')
ax2.plot(zs, util.Util.pdf_to_cdf(pdf_clr), color='tab:green', label='CLR')
ax2.set_ylabel("cdf")
ax2.legend()

# is_win = b_train_origin > z_train_origin
# is_lose = np.logical_not(is_win)
# T = z_train_origin
# T[is_lose] = b_train_origin[is_lose]

# kmf = KaplanMeierFitter()
# kmf.fit(T, event_observed=is_win, label='KMF', timeline=zs)  # or, more succinctly, kmf.fit(T, E)
# kmf_cdf = kmf.cumulative_density_['KMF'].tolist()
# kmf_pdf = util.Util.cdf_to_pdf(kmf_cdf)
# kmf_cdf = util.Util.pdf_to_cdf(kmf_pdf)
# ax1.plot(zs, kmf_pdf, color='tab:green', label='KMF')
# ax2.plot(zs, kmf_cdf, color='tab:green', label='KMF')

plt.tight_layout()

plt.savefig(sys.argv[1] + '/baselines' + ".pdf", format='pdf')

print(report)

df = pd.DataFrame(data=report._rows, columns=report.field_names)
output_path = '../result/baseline_report.csv'
df.to_csv(output_path, mode='a', index=False, sep='\t', header=not os.path.exists(output_path))

pickle.dump(km.pdf, open(sys.argv[1] + '/KM_pdf', 'wb'))
pickle.dump(pdf_cgmm, open(sys.argv[1] + '/CGMM_pdf', 'wb'))
pickle.dump(pdf_kmdt, open(sys.argv[1] + '/KMDT_pdf', 'wb'))
pickle.dump(pdf_clr, open(sys.argv[1] + '/CLR_pdf', 'wb'))
