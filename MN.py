from __future__ import print_function
import torch

torch.set_printoptions(edgeitems=5)

from torch.nn.functional import normalize
from torch.utils.tensorboard import SummaryWriter

import sys
import math
import random
import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

# eps = sys.float_info.epsilon
eps = 1e-8

class Util():
    def __init__(self):
        pass

    @staticmethod
    def generate_sparse(raw, value, dimension=300, batch_size=4096):
        """
        :param raw: raw log, such as b_train and z_train, which are both record_size * 1 matrix
        :param dimension:  dimension of market price or bid price
        :param batch_size:
        :return:
        """
        (record_size, _) = np.shape(raw)
        r = csr_matrix((dimension, 0), dtype=np.int8)

        starts, ends = Util.generate_batch_index(record_size, batch_size)

        for start, end in list(zip(starts, ends)):
            size = end - start
            batch = csr_matrix(
                (value[start: end], (raw[start: end].reshape(-1, ), np.arange(size))),
                shape=(dimension, size), dtype=np.int8)
            r = hstack((r, batch))
        return r.tocsr()

    @staticmethod
    def generate_sparse_b(b_origin, z_origin, is_lose, is_win, b_dimension=300):
        """
        generate new b to make MN more robust
        for winning case, new_b could be all possible values that satisfies new_b>=z+1
        for losing case, new_b could be all possible values that satisfies new_b<=z
        :param b_origin:
        :param z_origin:
        :param is_lose:
        :param dimension:
        :return:
        """

        (record_size, _) = np.shape(b_origin)
        win_data_size = b_dimension*is_win.sum()-(z_origin[is_win]+1).sum()
        lose_data_size = (z_origin[is_lose]+1).sum()
        # data = np.ones((win_data_size+lose_data_size,), dtype=np.int8)
        data = []
        row_ind = []
        col_ind = []

        for i in range(record_size):
            z = z_origin[i, 0]
            this_row_ind = list(range(z + 1)) if is_lose[i] else list(range(z+1, b_dimension))
            row_ind.extend(this_row_ind)
            col_ind.extend([i]*len(this_row_ind))
            data.extend([1.0/len(this_row_ind)]*len(this_row_ind))

        r = csr_matrix(
            (np.array(data).reshape(-1, ), (np.array(row_ind).reshape(-1, ), np.array(col_ind).reshape(-1, ))),
            shape=(b_dimension, record_size), dtype=np.float)

        return r.tocsr()

    @staticmethod
    def scipy_csr_2_torch_coo(csr, device, dtype):
        coo = csr.tocoo()
        return torch.sparse.FloatTensor(torch.LongTensor([coo.row, coo.col]),
                                        torch.FloatTensor(coo.data),
                                        torch.Size(coo.shape)).type(dtype).to(device)

    @staticmethod
    def probability_sampling(probabilities, device, dtype):
        """
        Sampling according to the given probabilities
        :param probabilities: torch.tensor, each entry determines the probability of this entry to be chosen
        :return: identities identify whether the corresponding entry has been chosen
        """
        return torch.gt(probabilities,
                        torch.rand_like(probabilities).to(device)).type(dtype)

    @staticmethod
    def generate_batch_index(size, batch_size=4096):
        starts = [i * batch_size for i in range(int(math.ceil(size / batch_size)))]
        ends = [i * batch_size for i in range(1, int(math.ceil(size / batch_size)))]
        ends.append(size)

        return starts, ends

    @staticmethod
    def cdf_to_pdf(cdf):
        pdf = [cdf[i + 1] - cdf[i] if cdf[i + 1] - cdf[i] > 0 else 1e-6 for i in range(len(cdf) - 1)]
        pdf.insert(0, 0.0 + eps)
        # normalize
        pdf_sum = sum(pdf)

        return [p / pdf_sum for p in pdf]

    @staticmethod
    def pdf_to_cdf(pdf):
        # normalize
        pdf_sum = sum(pdf)
        pdf = [pdf[i] / pdf_sum if pdf[i] > 0 else 1e-6 for i in range(len(pdf))]

        cdf = [sum(pdf[0:i]) for i in range(1, len(pdf))]
        cdf.append(1.0)
        return cdf

    @staticmethod
    def count_pdf_cdf(z, z_lower_bound=0, z_upper_bound=300):
        # z is m x 1 numpy.ndarray
        max_price = 0
        number_record = 1  # in case of dividing zero
        pdf = {}
        cdf = {}

        for s in np.asarray(z).reshape((-1,)).tolist():
            price = float(s) + eps
            price_int = math.floor(price)
            price_int = price_int if price_int < z_upper_bound else z_upper_bound
            max_price = price_int if max_price < price_int else max_price

            if price_int in pdf:
                pdf[price_int] += 1
            else:
                pdf[price_int] = 1

            # self.data.append(price_int)
            number_record += 1

        for price in range(z_lower_bound, z_upper_bound + 1):
            if price not in pdf:
                pdf[price] = 0

        for price in pdf:
            pdf[price] = pdf[price] / number_record

        for price in pdf:
            p = 0
            for j in pdf:
                p += pdf[j] if j <= price else 0
            cdf[price] = p

        return pdf, cdf


class Markov_Network():
    def __init__(self, x_dimension, encoder, root_path, fixed_w_b_z=False, fixed_w_x_omega=False,
                 fixed_c_b=False, fixed_c_z=False, fixed_c_x=False, fixed_c_omega=False,
                 z_dimension=300, b_dimension=300, dtype=torch.float32, device=torch.device("cpu")):
        self.x_dimension = x_dimension
        self.z_dimension = z_dimension
        self.b_dimension = b_dimension
        self.omega_dimension = 1
        self.encoder = encoder
        self.fixed_w_b_z = fixed_w_b_z
        self.fixed_w_x_omega = fixed_w_x_omega
        self.fixed_c_b = fixed_c_b
        self.fixed_c_z = fixed_c_z
        self.fixed_c_x = fixed_c_x
        self.fixed_c_omega = fixed_c_omega
        self.dtype = dtype
        self.device = device
        self.root_path = root_path
        self.writer = SummaryWriter('runs/' + root_path.split("/")[-1])

        self.w_b_z = torch.zeros(b_dimension, z_dimension, dtype=dtype, device=device)#.normal_(std=0.1)
        if fixed_w_b_z:
            for i in range(b_dimension):
                for j in range(z_dimension):
                    if i > j:
                        self.w_b_z[i, j] = 1
                    elif i < j:
                        self.w_b_z[i, j] = -1
                    else:
                        self.w_b_z[i, j] = -1

        self.w_b_omega = torch.zeros(b_dimension, 1, dtype=dtype, device=device)#.normal_(std=0.1)
        self.w_x_z = torch.zeros(x_dimension, z_dimension, dtype=dtype, device=device)#.normal_(std=0.1)
        self.w_x_omega = torch.zeros(x_dimension, 1, dtype=dtype, device=device)#.normal_(std=0.1)

        self.c_b = torch.zeros(b_dimension, 1, dtype=dtype, device=device)#.normal_(std=0.1)
        self.c_x = torch.zeros(x_dimension, 1, dtype=dtype, device=device)#.normal_(std=0.1)
        self.c_z = torch.zeros(z_dimension, 1, dtype=dtype, device=device)#.normal_(std=0.1)
        self.c_omega = torch.zeros(1, 1, dtype=dtype, device=device)#.normal_(std=0.1)

        # init x-feat range, which indicates the range of index for each feature in x
        # self.x_feat_range = {}  # key is feature's origin_index, value is feature's encoded index list
        # for key, value in encoder.feat.items():
        #     origin_index = key.split(':')[0]
        #     if origin_index not in self.x_feat_range:
        #         self.x_feat_range[origin_index] = [value]
        #     else:
        #         self.x_feat_range[origin_index].append(value)

    def _validate_b_dimension(self, b):
        (b_dimension, _) = b.size()
        assert b_dimension == self.b_dimension, \
            "bid price's dimension {0:d} must be equal to {1:d}".format(b_dimension, self.b_dimension)

    def _validate_x_dimension(self, x):
        (x_dimension, _) = x.size()
        assert x_dimension == self.x_dimension, \
            "dimension of bid request's feature {0:d} must be equal to {1:d}".format(x_dimension, self.x_dimension)

    def _validate_z_dimension(self, z):
        (z_dimension, _) = z.size()
        assert z_dimension == self.z_dimension, \
            "market price's dimension {0:d} must be equal to {1:d}".format(z_dimension, self.z_dimension)

    def _validate_omega_dimension(self, omega):
        (omega_dimension, _) = omega.size()
        assert omega_dimension == self.omega_dimension, \
            "omega's dimension {0:d} must be equal to {1:d}".format(omega_dimension, self.omega_dimension)

    def _validate_all(self, z, omega, b, x):
        self._validate_z_dimension(z)
        self._validate_omega_dimension(omega)
        self._validate_b_dimension(b)
        self._validate_x_dimension(x)

        (_, z_size) = z.size()
        (_, omega_size) = omega.size()
        (_, b_size) = b.size()
        (_, x_size) = x.size()

        assert z_size == omega_size == b_size == x_size, \
            "market prices' size {0:d}, omegas' size{1:d}, bid prices' size {2:d} and bid requests' size{3:d}" \
            "must be all equal".format(z_size, omega_size, b_size, x_size)

    def fit(self, z_origin, b_origin, x_origin, sample_rate=1.0, epoch=10, batch_size=512, scale=1, burn_in_epoch=0,
            lr_w_p=1e-1, lr_w_n=1e-1, lr_c_p=1e-1, lr_c_n=1e-1, l2_decay=1e-2, verbose=1, debug_interval=256):
        """
        :param z_origin: record_num * 1 numpy.ndarray
        :param b_origin: record_num * 1 numpy.ndarray
        :param x_origin: record_num * x_dimension scipy.sparse.csr_matrix
        :param sample_rate:
        :param epoch:
        :param batch_size:
        :param scale:
        :param burn_in_epoch:
        :param lr_w_p:
        :param lr_w_n:
        :param lr_c_p:
        :param lr_c_n:
        :param l2_decay:
        :param verbose:
        :return:
        """
        (z_size, _) = np.shape(z_origin)
        (b_size, _) = np.shape(b_origin)
        (x_size, _) = np.shape(x_origin)

        assert z_size == b_size == x_size, \
            "market prices' size {0:d}, bid prices' size {1:d} and bid requests' size{2:d}" \
            "must be all equal".format(z_size, b_size, x_size)

        global_step = 0

        print("Now begin to pre-process dataset")
        # z: z_dimension * record_num scipy.sparse.csr_matrix
        # omega: 1 * record_num numpy.ndarray
        # b: b_dimension * record_num scipy.sparse.csr_matrix
        # x: x_dimension * record_num scipy.sparse.csr_matrix
        print("Generating x")
        x = x_origin.transpose()
        is_lose = z_origin >= b_origin
        is_win = z_origin < b_origin
        omega = is_lose.astype(np.int8).transpose()

        lose = np.sum(omega)
        win = x_size - lose
        print("Total records: {0:d}, winning records: {1:d}, losing records: {2:d}"
              .format(x_size, win, lose))
        self.writer.add_scalar('datasets/total_records', x_size, 0)
        self.writer.add_scalar('datasets/winning records', win, 0)
        self.writer.add_scalar('datasets/losing records', lose, 0)
        self.writer.add_scalar('datasets/winning rate', win/x_size, 0)

        print("Generating b")
        b = Util.generate_sparse(b_origin, np.ones((b_size,), dtype=np.int8), self.b_dimension, 16384)
        # b = Util.generate_sparse_b(b_origin, z_origin, is_lose, is_win, self.b_dimension)
        # count prior of b
        self.b_prior = np.zeros((self.b_dimension,)) + eps
        unique, counts = np.unique(b_origin, return_counts=True)
        for k, v in zip(unique, counts):
            self.b_prior[int(k)] = v / b_size

        print("Generating censored z")
        z_value = np.ones((z_size,), dtype=np.int8)
        z_value[is_lose.reshape(-1, )] = 0
        z = Util.generate_sparse(z_origin, z_value, self.z_dimension, 16384)
        z_value = None
        # print("Censoring z")
        # z[:, is_lose.reshape(-1, )] = 0

        print("Now begin to fit, hyper-parameters: "
              "sample_rate:{6:.2f}, epoch:{0:d}, batch_size:{1:d}, scale:{7:d}, burn_in_epoch:{8:d}, "
              "lr_w_p:{2:f}, lr_w_n:{3:f}, lr_c_p:{4:f}, lr_c_n:{5:f}"
              .format(epoch, batch_size, lr_w_p, lr_w_n, lr_c_p, lr_c_n, sample_rate, scale, burn_in_epoch))

        for i in range(epoch):  # in an epoch
            this_lr_w_p = lr_w_p / math.sqrt(i + 1)
            this_lr_w_n = lr_w_n / math.sqrt(i + 1)
            this_lr_c_p = lr_c_p / math.sqrt(i + 1)
            this_lr_c_n = lr_c_n / math.sqrt(i + 1)
            this_l2_decay = l2_decay / math.sqrt(i + 1)
            # this_sample_rate = sample_rate / math.sqrt(i + 1)

            mask = np.random.choice([False, True], z_size, p=[1 - sample_rate, sample_rate])
            sample_size = z[:, mask].shape[1]
            if verbose == 1:
                print("============== epoch: {} ==============".format(str(i)))
                print("{0:d} records have been sampled".format(sample_size))
                print(", market prices' shape: ", z[:, mask].shape,
                      ", omegas' shape: ", omega[:, mask].shape,
                      ", bid prices' shape: ", b[:, mask].shape,
                      ", bid requests' shape: ", x[:, mask].shape, )

            # generate new b to make MN more robust
            # for winning case omega==0 new_b = z + alpha*(b_max-z)
            # for losing case omega==1  new_b = z - alpha*(z-z_min)
            # new_b = np.zeros_like(b_origin[mask, :])
            # this_lose = np.sum(omega[:, mask])
            # this_win = sample_size - this_lose
            #
            # alpha = np.random.randint(0, 101, size=(sample_size, 1)) / 100
            # b_max = np.ones((this_win, 1), dtype=np.int16) * (self.b_dimension - 1)
            # z_min = np.ones((this_lose, 1), dtype=np.int16) * 0
            #
            # new_b[is_win[mask].reshape(-1, ), :] = np.ceil(z_origin[mask, :][is_win[mask].reshape(-1, ), :] + \
            #                                  alpha[is_win[mask].reshape(-1, ), :] * (
            #                                              b_max - z_origin[mask, :][is_win[mask].reshape(-1, ), :]))
            # new_b[is_lose[mask].reshape(-1, ), :] = np.floor(z_origin[mask, :][is_lose[mask].reshape(-1, ), :] - \
            #                                   alpha[is_lose[mask].reshape(-1, ), :] * (
            #                                               z_origin[mask, :][is_lose[mask].reshape(-1, ), :] - z_min))
            #
            # this_new_b = Util.generate_sparse(new_b, np.ones((sample_size,), dtype=np.int8), self.b_dimension, 16384)

            # mini-batch gradient descend
            starts, ends = Util.generate_batch_index(sample_size, batch_size)
            index = []
            index.extend(list(zip(starts, ends)))
            random.shuffle(index)

            for start, end in index:  # in a mini-batch
                this_batch_size = end - start
                this_z = Util.scipy_csr_2_torch_coo(z[:, mask][:, start:end], self.device, self.dtype).to_dense()
                this_omega = torch.from_numpy(omega[:, mask][:, start:end]).type(self.dtype).to(self.device)
                this_b = Util.scipy_csr_2_torch_coo(b[:, mask][:, start:end], self.device, self.dtype).to_dense()
                this_x = Util.scipy_csr_2_torch_coo(x[:, mask][:, start:end], self.device, self.dtype).to_dense()

                # for the losing cases, generate z ~ p(z|b,x)
                # this_is_lose = (z_origin[mask, :][start:end, :] > b_origin[mask, :][start:end, :]).reshape(-1, )
                # this_is_win = (z_origin[mask, :][start:end, :] <= b_origin[mask, :][start:end, :]).reshape(-1, )
                # new_original_z = np.zeros((this_batch_size, 1), dtype=np.int16)
                # new_original_z[this_is_lose, :] = \
                #     self.generate_original_z_given_b_x(this_b[:, this_is_lose], this_x[:, this_is_lose],
                #                                 b_origin[mask, :][start:end, :][this_is_lose, :]).cpu().reshape(-1, 1)
                # new_original_z[this_is_win, :] = z_origin[mask, :][start:end, :][this_is_win, :]
                #
                # this_z = Util.scipy_csr_2_torch_coo(
                #     Util.generate_sparse(new_original_z, np.ones((this_batch_size,), dtype=np.int8), self.z_dimension),
                #     self.device, self.dtype).to_dense()

                # print("z:", this_z)
                # print("omega:", this_omega)
                # print("b:", this_b)
                # print("x:", this_x)

                # derivatives for positive phase
                pos_der_w_b_z = this_b.mm(this_z.t()) / this_batch_size
                pos_der_w_b_omega = this_b.mm(this_omega.t()) / this_batch_size
                pos_der_w_x_z = this_x.mm(this_z.t()) / this_batch_size
                pos_der_w_x_omega = this_x.mm(this_omega.t()) / this_batch_size
                pos_der_c_z = this_z.sum(dim=1).view(-1, 1) / this_batch_size
                pos_der_c_omega = this_omega.sum(dim=1).view(-1, 1) / this_batch_size
                pos_der_c_b = this_b.sum(dim=1).view(-1, 1) / this_batch_size
                pos_der_c_x = this_x.sum(dim=1).view(-1, 1) / this_batch_size

                # derivatives for negative phase
                # all negative samples are dense matrices
                z_samples, omega_samples, \
                b_samples, x_samples = self.sample(this_z, this_omega, this_b, this_x,
                                                   scale=scale, burn_in_epoch=burn_in_epoch)
                (_, negative_size) = z_samples.size()

                neg_der_w_b_z = b_samples.mm(z_samples.t()) / negative_size
                neg_der_w_b_omega = b_samples.mm(omega_samples.t()) / negative_size
                neg_der_w_x_z = x_samples.mm(z_samples.t()) / negative_size
                neg_der_w_x_omega = x_samples.mm(omega_samples.t()) / negative_size
                neg_der_c_z = z_samples.sum(dim=1).view(-1, 1) / negative_size
                neg_der_c_omega = omega_samples.sum(dim=1).view(-1, 1) / negative_size
                neg_der_c_b = b_samples.sum(dim=1).view(-1, 1) / negative_size
                neg_der_c_x = x_samples.sum(dim=1).view(-1, 1) / negative_size

                def write_heatmap(data, tag):
                    f, ax = plt.subplots(1, 1, sharex=True)
                    sns.heatmap(data, ax=ax)
                    self.writer.add_figure(tag, f)
                    plt.close(f)

                if global_step % debug_interval == 0:
                    sum_pos_der_w_b_z = torch.sum(pos_der_w_b_z)
                    sum_neg_der_w_b_z = torch.sum(neg_der_w_b_z)
                    sum_pos_der_w_b_omega = torch.sum(pos_der_w_b_omega)
                    sum_neg_der_w_b_omega = torch.sum(neg_der_w_b_omega)
                    sum_pos_der_w_x_z = torch.sum(pos_der_w_x_z)
                    sum_neg_der_w_x_z = torch.sum(neg_der_w_x_z)
                    sum_pos_der_w_x_omega = torch.sum(pos_der_w_x_omega)
                    sum_neg_der_w_x_omega = torch.sum(neg_der_w_x_omega)

                    sum_pos_der_c_z = torch.sum(pos_der_c_z)
                    sum_neg_der_c_z = torch.sum(neg_der_c_z)
                    sum_pos_der_c_omega = torch.sum(pos_der_c_omega)
                    sum_neg_der_c_omega = torch.sum(neg_der_c_omega)
                    sum_pos_der_c_b = torch.sum(pos_der_c_b)
                    sum_neg_der_c_b = torch.sum(neg_der_c_b)
                    sum_pos_der_c_x = torch.sum(pos_der_c_x)
                    sum_neg_der_c_x = torch.sum(neg_der_c_x)

                    print("========DEBUG INFO========")
                    print("positive derivative sum", "negative derivative sum")
                    print("der_w_b_z", sum_pos_der_w_b_z, sum_neg_der_w_b_z)
                    print("der_w_b_omega", sum_pos_der_w_b_omega, sum_neg_der_w_b_omega)
                    print("der_w_x_z", sum_pos_der_w_x_z, sum_neg_der_w_x_z)
                    print("der_w_x_omega", sum_pos_der_w_x_omega, sum_neg_der_w_x_omega)
                    print("der_c_z", sum_pos_der_c_z, sum_neg_der_c_z)
                    print("der_c_omega", sum_pos_der_c_omega, sum_neg_der_c_omega)
                    print("der_c_b", sum_pos_der_c_b, sum_neg_der_c_b)
                    print("der_c_x", sum_pos_der_c_x, sum_neg_der_c_x)

                    debug_step = int(global_step / debug_interval)

                    self.writer.add_scalar('train/der_sum/w_b_z/positive', sum_pos_der_w_b_z, debug_step)
                    self.writer.add_scalar('train/der_sum/w_b_z/negative', sum_neg_der_w_b_z, debug_step)
                    self.writer.add_scalar('train/der_sum/w_b_omega/positive', sum_pos_der_w_b_omega, debug_step)
                    self.writer.add_scalar('train/der_sum/w_b_omega/negative', sum_neg_der_w_b_omega, debug_step)
                    self.writer.add_scalar('train/der_sum/w_x_z/positive', sum_pos_der_w_x_z, debug_step)
                    self.writer.add_scalar('train/der_sum/w_x_z/negative', sum_neg_der_w_x_z, debug_step)
                    self.writer.add_scalar('train/der_sum/w_x_omega/positive', sum_pos_der_w_x_omega, debug_step)
                    self.writer.add_scalar('train/der_sum/w_x_omega/negative', sum_neg_der_w_x_omega, debug_step)
                    self.writer.add_scalar('train/der_sum/c_z/positive', sum_pos_der_c_z, debug_step)
                    self.writer.add_scalar('train/der_sum/c_z/negative', sum_neg_der_c_z, debug_step)
                    self.writer.add_scalar('train/der_sum/c_omega/positive', sum_pos_der_c_omega, debug_step)
                    self.writer.add_scalar('train/der_sum/c_omega/negative', sum_neg_der_c_omega, debug_step)
                    self.writer.add_scalar('train/der_sum/c_b/positive', sum_pos_der_c_b, debug_step)
                    self.writer.add_scalar('train/der_sum/c_b/negative', sum_neg_der_c_b, debug_step)
                    self.writer.add_scalar('train/der_sum/c_x/positive', sum_pos_der_c_x, debug_step)
                    self.writer.add_scalar('train/der_sum/c_x/negative', sum_neg_der_c_x, debug_step)

                    write_heatmap(pos_der_w_b_z.cpu().numpy(), "ders/pos/w_b_z")
                    write_heatmap(neg_der_w_b_z.cpu().numpy(), "ders/neg/w_b_z")
                    write_heatmap(pos_der_c_z.cpu().numpy(), "ders/pos/c_z")
                    write_heatmap(neg_der_c_z.cpu().numpy(), "ders/neg/c_z")
                    write_heatmap(pos_der_c_b.cpu().numpy(), "ders/pos/c_b")
                    write_heatmap(neg_der_c_b.cpu().numpy(), "ders/neg/c_b")

                global_step += 1

                if not self.fixed_w_b_z:
                    self.w_b_z = self.w_b_z + (this_lr_w_p * pos_der_w_b_z - this_lr_w_n * neg_der_w_b_z) \
                                 - self.w_b_z * this_l2_decay
                self.w_b_omega = self.w_b_omega + (this_lr_w_p * pos_der_w_b_omega - this_lr_w_n * neg_der_w_b_omega) \
                                 - self.w_b_omega * this_l2_decay
                self.w_x_z = self.w_x_z + (this_lr_w_p * pos_der_w_x_z - this_lr_w_n * neg_der_w_x_z) \
                             - self.w_x_z * this_l2_decay
                if not self.fixed_w_x_omega:
                    self.w_x_omega = self.w_x_omega + (
                            this_lr_w_p * pos_der_w_x_omega - this_lr_w_n * neg_der_w_x_omega) \
                                     - self.w_x_omega * this_l2_decay
                if not self.fixed_c_b:
                    self.c_b = self.c_b + (this_lr_c_p * pos_der_c_b - this_lr_c_n * neg_der_c_b) \
                               - self.c_b * this_l2_decay
                if not self.fixed_c_x:
                    self.c_x = self.c_x + (this_lr_c_p * pos_der_c_x - this_lr_c_n * neg_der_c_x) \
                                   - self.c_x * this_l2_decay
                if not self.fixed_c_omega:
                    self.c_omega = self.c_omega + (this_lr_c_p * pos_der_c_omega - this_lr_c_n * neg_der_c_omega) \
                                   - self.c_omega * this_l2_decay
                if not self.fixed_c_z:
                    self.c_z = self.c_z + (this_lr_c_p * pos_der_c_z - this_lr_c_n * neg_der_c_z) \
                               - self.c_z * this_l2_decay

            KL_pdf, WD_pdf, KL_cdf, WD_cdf, ANLP, MSE, O, c_index = \
                self.evaluate(x[:, mask], z_origin[mask, :], b=b[:, mask], omega_for_z=omega[:, mask], epoch=i,
                              path=self.root_path + "/evaluate/{6:.2f}_{0:d}_{1:d}_{7:d}_{8:d}_{10:.6f}_{2:.3f}_{3:.3f}_{4:.3f}_{5:.3f}"
                              .format(epoch, batch_size, lr_w_p, lr_w_n, lr_c_p, lr_c_n, sample_rate, scale,
                                      burn_in_epoch, i, l2_decay))

            self.writer.add_scalar('train/pdf/KL', KL_pdf, i)
            self.writer.add_scalar('train/pdf/WD', WD_pdf, i)
            self.writer.add_scalar('train/cdf/KL', KL_cdf, i)
            self.writer.add_scalar('train/cdf/WD', WD_cdf, i)
            self.writer.add_scalar('train/MSE', MSE, i)
            self.writer.add_scalar('train/ANLP', ANLP, i)
            self.writer.add_scalar('train/c_index', c_index, i)
            self.writer.add_scalar('train/omega', O, i)

            write_heatmap(self.w_b_z.cpu().numpy(), "paras/w_b_z")
            write_heatmap(self.w_b_omega.cpu().numpy(), "paras/w_b_omega")
            write_heatmap(self.c_b.cpu().numpy(), "paras/c_b")
            write_heatmap(self.c_z.cpu().numpy(), "paras/c_z")

    # def ANLP_overall(self, z, x, batch_size=4096):
    #     """
    #     Compute the averaged log likelihood of given samples.
    #     It is intractable to compute the joint likelihood of given samples according to this Markov Network
    #     since we do not work out the normalized joint p.d.f.
    #     Though, we can compute the conditional likelihood of z and omega for a given b and x.
    #
    #     :param z: record_num * 1
    #     :type z: numpy.ndarray
    #     :param omega: 1 * record_num
    #     :type omega: numpy.ndarray
    #     :param b: record_num * 1
    #     :type b: numpy.ndarray
    #     :param batch_size:
    #     :param x: x_dimension * record_num
    #     :type x: scipy.sparse.csr_matrix
    #     :return:
    #     """
    #
    #     (z_size, _) = z.shape
    #     (_, x_size) = x.shape
    #     assert z_size == x_size, \
    #         "Market prices' size {0:d} must be equal to bid requests' size{1:d}".format(z_size, x_size)
    #
    #     if b is None:
    #         b = np.zeros((x_size, 1))
    #         # bid the maximal price
    #         b[:, :] = self.b_dimension - 1
    #
    #     ll = torch.zeros(1, 1, dtype=self.dtype, device=self.device).view(-1, )
    #     pdf_unnormalized = torch.zeros(1, 1, dtype=self.dtype, device=self.device).view(-1, )
    #
    #     starts, ends = Util.generate_batch_index(x_size, batch_size)
    #
    #     for start, end in list(zip(starts, ends)):
    #         this_size = end - start
    #         this_z = Util.scipy_csr_2_torch_coo(
    #             Util.generate_sparse(z[start:end, :], np.ones((this_size,), dtype=np.int8), self.z_dimension),
    #             self.device, self.dtype).to_dense()
    #         this_x = Util.scipy_csr_2_torch_coo(x[:, start:end], self.device, self.dtype)
    #
    #         this_ll, this_pdf = self.log_likelihood(this_z, this_omega, this_x, this_b)
    #         ll += this_ll.sum()
    #         pdf_unnormalized += this_pdf.sum()
    #
    #     return -ll[0] / x_size, pdf_unnormalized[0] / x_size
    #
    # def averaged_log_probability_hat(self, z, omega, b, x):
    #     self._validate_all(z, omega, b, x)
    #     (_, dataset_size) = z.size()
    #     all = torch.sum(b.t().mm(self.w_b_z).mul(z.t()) +
    #                     b.t().mm(self.w_b_omega).mul(omega.t()) +
    #                     x.t().mm(self.w_x_z).mul(z.t()) +
    #                     x.t().mm(self.w_x_omega).mul(omega.t())) + \
    #           torch.sum(z.t().mm(self.c_z) + omega.t().mm(self.c_omega) +
    #                     b.t().mm(self.c_b) + x.t().mm(self.c_x))
    #
    #     return all / dataset_size
    #
    # def log_likelihood(self, z, omega, x, b):
    #     # pdf_hat, omega_hat = self.softmax_probability_z_omega_given_b_x(b, x)
    #     pdf_hat, omega_hat = self.probability_z_omega_given_b_x(b, x)
    #     return (z.mul(pdf_hat).sum(dim=0).view(1, -1) +
    #             omega.mul(omega_hat).view(1, -1) + eps).log(), pdf_hat

    def _normalize_negative_samples(self, z_hat, omega_hat, b_hat, x_hat):
        return z_hat, omega_hat, b_hat, x_hat

    def sample(self, z, omega, b, x, scale=1, burn_in_epoch=0):
        """
        Sample from our Markov_Network.
        Since it is hard to generate samples directly,
        we use Gibbs sampling according to p(z,omega|b,x) and p(b,x|z,omega).

        :param z: the initial state of Markov Chain
        :param omega: the initial state of Markov Chain
        :param b: the initial state of Markov Chain
        :param x: the initial state of Markov Chain
        :param scale:the number of samples we want to make
        :param epoch: the number of burn-in epochs
        """
        self._validate_all(z, omega, b, x)
        (_, dataset_size) = z.size()

        z_samples = torch.zeros(self.z_dimension, 0, dtype=self.dtype, device=self.device)
        omega_samples = torch.zeros(self.omega_dimension, 0, dtype=self.dtype, device=self.device)
        b_samples = torch.zeros(self.b_dimension, 0, dtype=self.dtype, device=self.device)
        x_samples = torch.zeros(self.x_dimension, 0, dtype=self.dtype, device=self.device)

        # initialize samples
        z_hat = z
        omega_hat = omega
        b_hat = b
        x_hat = x

        # z_hat = torch.zeros(self.z_dimension, dataset_size, dtype=self.dtype, device=self.device).normal_(std=0.1)
        # omega_hat = torch.zeros(self.omega_dimension, dataset_size, dtype=self.dtype, device=self.device).normal_(std=0.1)
        # b_hat = torch.zeros(self.b_dimension, dataset_size, dtype=self.dtype, device=self.device).normal_(std=0.1)
        # x_hat = torch.zeros(self.x_dimension, dataset_size, dtype=self.dtype, device=self.device).normal_(std=0.1)

        for i in range(burn_in_epoch):
            # We should discard all samples in burn-in epochs
            z_hat, omega_hat, b_hat, x_hat = self.Gibbs_sampling_one_step(z_hat, omega_hat, b_hat, x_hat)

        for i in range(scale):
            z_hat, omega_hat, b_hat, x_hat = self.Gibbs_sampling_one_step(z_hat, omega_hat, b_hat, x_hat)

            z_samples = torch.cat((z_samples, z_hat), dim=1)
            omega_samples = torch.cat((omega_samples, omega_hat), dim=1)
            b_samples = torch.cat((b_samples, b_hat), dim=1)
            x_samples = torch.cat((x_samples, x_hat), dim=1)

        return self._normalize_negative_samples(z_samples, omega_samples, b_samples, x_samples)

    def Gibbs_sampling_one_step(self, z, omega, b, x):
        # z omega b x
        # probability_z, probability_omega = self.probability_z_omega_given_b_x(b, x)
        # z_hat = Util.probability_sampling(probability_z, self.device, self.dtype)
        # omega_hat = Util.probability_sampling(probability_omega, self.device, self.dtype)
        #
        # probability_b, probability_x = self.probability_b_x_given_z_omega(z_hat, omega_hat)
        # b_hat = Util.probability_sampling(probability_b, self.device, self.dtype)
        # x_hat = Util.probability_sampling(probability_x, self.device, self.dtype)
        #
        # return z_hat, omega_hat, b_hat, x_hat

        # b x z omega
        # probability_b, probability_x = self.probability_b_x_given_z_omega(z, omega)
        # b_hat = Util.probability_sampling(probability_b, self.device, self.dtype)
        # x_hat = Util.probability_sampling(probability_x, self.device, self.dtype)
        #
        # probability_z, probability_omega = self.probability_z_omega_given_b_x(b_hat, x_hat)
        # z_hat = Util.probability_sampling(probability_z, self.device, self.dtype)
        # omega_hat = Util.probability_sampling(probability_omega, self.device, self.dtype)
        #
        # return z_hat, omega_hat, b_hat, x_hat

        # combined sampling
        # probability_z, probability_omega = self.probability_z_omega_given_b_x(b, x)
        # z_hat = Util.probability_sampling(probability_z, self.device, self.dtype)
        # omega_hat = Util.probability_sampling(probability_omega, self.device, self.dtype)
        #
        # probability_b, probability_x = self.probability_b_x_given_z_omega(z, omega)
        # b_hat = Util.probability_sampling(probability_b, self.device, self.dtype)
        # x_hat = Util.probability_sampling(probability_x, self.device, self.dtype)
        #
        # return z_hat, omega_hat, b_hat, x_hat

        # original combined sampling
        # z_hat, omega_hat = self.probability_z_omega_given_b_x(b, x)
        # b_hat, x_hat = self.probability_b_x_given_z_omega(z, omega)
        #
        # return z_hat, omega_hat, b_hat, x_hat

        # double sampling : p(b,x|z,omega)p(z,omega|b,x) + p(z,omega|b,x)p(b,x|z,omega)
        # probability_b, probability_x = self.probability_b_x_given_z_omega(z, omega)
        # b_hat1 = Util.probability_sampling(probability_b, self.device, self.dtype)
        # x_hat1 = Util.probability_sampling(probability_x, self.device, self.dtype)
        #
        # probability_z, probability_omega = self.probability_z_omega_given_b_x(b_hat1, x_hat1)
        # z_hat1 = Util.probability_sampling(probability_z, self.device, self.dtype)
        # omega_hat1 = Util.probability_sampling(probability_omega, self.device, self.dtype)
        #
        # probability_z, probability_omega = self.probability_z_omega_given_b_x(b, x)
        # z_hat2 = Util.probability_sampling(probability_z, self.device, self.dtype)
        # omega_hat2 = Util.probability_sampling(probability_omega, self.device, self.dtype)
        #
        # probability_b, probability_x = self.probability_b_x_given_z_omega(z_hat2, omega_hat2)
        # b_hat2 = Util.probability_sampling(probability_b, self.device, self.dtype)
        # x_hat2 = Util.probability_sampling(probability_x, self.device, self.dtype)
        #
        # return torch.cat((z_hat1, z_hat2), dim=1), \
        #        torch.cat((omega_hat1, omega_hat2), dim=1), \
        #        torch.cat((b_hat1, b_hat2), dim=1), \
        #        torch.cat((x_hat1, x_hat2), dim=1)
        # return (z_hat1 + z_hat2) / 2, (omega_hat1 + omega_hat2) / 2, (b_hat1 + b_hat2) / 2, (x_hat1 + x_hat2) / 2

        # original double sampling
        b_hat1, x_hat1 = self.probability_b_x_given_z_omega(z, omega)
        z_hat1, omega_hat1 = self.probability_z_omega_given_b_x(b_hat1, x_hat1)

        z_hat2, omega_hat2 = self.probability_z_omega_given_b_x(b, x)
        b_hat2, x_hat2 = self.probability_b_x_given_z_omega(z_hat2, omega_hat2)

        # return torch.cat((z_hat1, z_hat2), dim=1), \
        #        torch.cat((omega_hat1, omega_hat2), dim=1), \
        #        torch.cat((b_hat1, b_hat2), dim=1), \
        #        torch.cat((x_hat1, x_hat2), dim=1)
        return (z_hat1 + z_hat2) / 2, (omega_hat1 + omega_hat2) / 2, (b_hat1 + b_hat2) / 2, (x_hat1 + x_hat2) / 2

    def z_omega_compatible_b_x(self, b, x):
        (_, b_size) = b.size()
        return self.w_b_z.t().mm(b) + x.t().mm(self.w_x_z).t() + self.c_z.repeat(1, b_size), \
               self.w_b_omega.t().mm(b) + x.t().mm(self.w_x_omega).t() + self.c_omega.repeat(1, b_size)

    def b_x_compatible_z_omega(self, z, omega):
        (_, z_size) = z.size()
        return self.w_b_z.mm(z) + self.w_b_omega.mm(omega) + self.c_b.repeat(1, z_size), \
               self.w_x_z.mm(z) + self.w_x_omega.mm(omega) + self.c_x.repeat(1, z_size)

    def probability_z_omega_given_b_x(self, b, x):
        # self._validate_b_dimension(b)
        # self._validate_x_dimension(x)
        #
        # (_, b_size) = b.size()
        # (_, x_size) = x.size()
        # assert b_size == x_size, \
        #     "bid prices' size {0:d} must be equal to bid requests' size{1:d}".format(b_size, x_size)

        #  All dimensions of z and omega are independent to each other
        z_compatible_b_x, omega_compatible_b_x = self.z_omega_compatible_b_x(b, x)

        return torch.sigmoid(z_compatible_b_x), torch.sigmoid(omega_compatible_b_x)

    def softmax_probability_z_omega_given_b_x(self, b, x):
        """
        :param b: b_dimension * record_num dense matrix
        :param x: x_dimension * record_num torch.sparse.coo_matrix
        :return:
        """
        # self._validate_b_dimension(b)
        # self._validate_x_dimension(x)
        #
        # (_, b_size) = b.size()
        # (_, x_size) = x.size()
        # assert b_size == x_size, \
        #     "bid prices' size {0:d} must be equal to bid requests' size{1:d}".format(b_size, x_size)

        z_compatible_b_x, omega_compatible_b_x = self.z_omega_compatible_b_x(b, x)

        # softmax = torch.softmax(torch.cat((z_compatible_b_x, omega_compatible_b_x), dim=0), dim=0)
        # return softmax[0:self.z_dimension, :], \
        #        softmax[self.z_dimension, :].view(1, -1)

        # temp = torch.cat((z_compatible_b_x, omega_compatible_b_x), dim=0)
        # r = torch.softmax(temp, dim=0)
        #
        # return r[0:-1, ], r[-1, ].reshape((1, -1))
        return torch.softmax(z_compatible_b_x, dim=0), torch.sigmoid(omega_compatible_b_x)

    def probability_b_x_given_z_omega(self, z, omega):
        self._validate_z_dimension(z)
        self._validate_omega_dimension(omega)

        (_, z_size) = z.size()
        (_, omega_size) = omega.size()
        assert z_size == omega_size, \
            "market prices' size {0:d} must be equal to omegas' size{1:d}".format(z_size, omega_size)

        #  All dimensions of b and x are independent to each other

        b_compatible_z_omega, x_compatible_z_omega = self.b_x_compatible_z_omega(z, omega)

        return torch.sigmoid(b_compatible_z_omega), torch.sigmoid(x_compatible_z_omega)

    # def generate_original_z_given_b_x(self, b, x, b_origin):
    #     """
    #     :param b:
    #     :param x:
    #     :return: original_z
    #     """
    #     (_, size) = b.size()
    #     pdf, _ = self.probability_z_omega_given_b_x(b, x)
    #     # TODO: need to speed up
    #     for i in range(size):
    #         pdf[0:int(b_origin[i, :]+1), i] = 0
    #
    #     return torch.argmax(pdf, dim=0)

    def predict_pdf_omega_individual_MN(self, x):
        """
        Predict the pdf of market price and omega for each instance of a given dataset x
        :param x: x_dimension * record_num
        :type x: torch.sparse.coo_matrix
        :return:
        """
        (_, x_size) = x.size()
        b = torch.zeros(self.b_dimension, x_size, dtype=self.dtype, device=self.device)
        # TODO: bid the maximal price or a random price?
        b_origin = np.random.randint(0, self.b_dimension, size=(x_size, 1))
        b[b_origin[:, 0].reshape(-1, ), np.array(list(range(x_size)))] = 1
        # b[-1, :] = 1

        return self.softmax_probability_z_omega_given_b_x(b, x)

    def predict_z_pdf_omega_overall_MN(self, x, z_origin, batch_size=8192, b_samples=1):
        """
        Predict the overall pdf for a given dataset x
        :param batch_size:
        :param x: x_dimension * record_num
        :type x: scipy.sparse.csr_matrix
        :param z_origin: record_num * 1
        :type z_origin: numpy.ndarray
        :return:
        """
        (_, x_size) = x.shape

        # Cut down memory usage by using batch prediction
        starts, ends = Util.generate_batch_index(x_size, batch_size)

        z = torch.zeros(x_size, 1, dtype=self.dtype, device=self.device).view(-1, )
        pdf = torch.zeros(self.z_dimension, 1, dtype=self.dtype, device=self.device).view(-1, )
        omega = torch.zeros(self.omega_dimension, 1, dtype=self.dtype, device=self.device).view(-1, )
        ll_z = torch.zeros(1, 1, dtype=self.dtype, device=self.device).view(-1, )

        for start, end in list(zip(starts, ends)):
            this_size = end - start
            this_z = Util.scipy_csr_2_torch_coo(
                Util.generate_sparse(z_origin[start:end, :], np.ones((this_size,), dtype=np.int8), self.z_dimension),
                self.device, self.dtype).to_dense()
            this_x = Util.scipy_csr_2_torch_coo(x[:, start:end], self.device, self.dtype)

            market_price_mask = torch.zeros(self.z_dimension, this_size, dtype=self.dtype, device=self.device)
            for i in range(self.z_dimension):
                market_price_mask[i, :] = i

            for j in range(b_samples):
                this_pdf, this_omega = self.predict_pdf_omega_individual_MN(this_x)
                this_ll_z = (this_z.mul(this_pdf).sum(dim=0).view(1, -1) + 1e-5).log()

                ll_z += this_ll_z.sum() / b_samples
                pdf += this_pdf.sum(dim=1).view(-1, ) / b_samples
                omega += this_omega.sum(dim=1).view(-1, ) / b_samples
                z[start:end] += this_pdf.mul(market_price_mask).sum(dim=0) / b_samples

        return z, pdf / x_size, omega / x_size, -ll_z[0] / x_size

    def anlp_z_omega_given_b_x(self, b, x, z_origin, omega, batch_size=8192):
        """
        :param b: b_dimension * record_num
        :type b: scipy.sparse.csr_matrix
        :param x: x_dimension * record_num
        :type x: scipy.sparse.csr_matrix
        :param z_origin: record_num * 1
        :type z_origin: numpy.ndarray
        :param omega: 1 * record_num
        :type omega: numpy.ndarray
        :param batch_size:
        :return:
        """
        (_, x_size) = x.shape
        starts, ends = Util.generate_batch_index(x_size, batch_size)

        ll_z = torch.zeros(1, 1, dtype=self.dtype, device=self.device).view(-1, )
        ll_omega = torch.zeros(1, 1, dtype=self.dtype, device=self.device).view(-1, )

        for start, end in list(zip(starts, ends)):
            this_size = end - start
            this_b = Util.scipy_csr_2_torch_coo(b[:, start:end], self.device, self.dtype).to_dense()
            this_x = Util.scipy_csr_2_torch_coo(x[:, start:end], self.device, self.dtype)
            this_z = Util.scipy_csr_2_torch_coo(
                Util.generate_sparse(z_origin[start:end, :], np.ones((this_size,), dtype=np.int8), self.z_dimension),
                self.device, self.dtype).to_dense()
            this_omega = torch.from_numpy(omega[:, start:end]).type(self.dtype).to(self.device)
            is_lose = (this_omega > 0.5).view(-1,)
            is_win = (this_omega < 0.5).view(-1,)

            this_pdf, this_lose = self.softmax_probability_z_omega_given_b_x(this_b, this_x)
            this_ll_z = (this_z.mul(this_pdf)[:, is_win].sum(dim=0).view(1, -1) + eps).log()

            this_ll_omega_lose = (this_omega.mul(this_lose)[:, is_lose].sum(dim=0).view(1, -1) + eps).log()
            this_ll_omega_win = ((1-this_omega).mul(1-this_lose)[:, is_win].sum(dim=0).view(1, -1) + eps).log()
            ll_z += this_ll_z.sum()
            ll_omega += this_ll_omega_lose.sum() + this_ll_omega_win.sum()

        return -ll_z[0] / (omega < 0.5).sum(), -ll_omega[0] / x_size

    # def predict_z(self, x, batch_size=4096):
    #     """
    #     Predict the market price for each instance of a given dataset x
    #     :param batch_size:
    #     :param x: x_dimension * record_num
    #     :type x: scipy.sparse.csr_matrix
    #     :return:
    #     """
    #     (_, x_size) = x.shape
    #
    #     # Cut down memory usage by using batch prediction
    #     starts, ends = Util.generate_batch_index(x_size, batch_size)
    #
    #     z = torch.zeros(x_size, 1, dtype=self.dtype, device=self.device).view(-1, )
    #     for start, end in list(zip(starts, ends)):
    #         this_size = end - start
    #         this_pdf, _ = self.predict_pdf_omega_individual(
    #             Util.scipy_csr_2_torch_coo(x[:, start:end], self.device, self.dtype))
    #         market_price = torch.zeros(self.z_dimension, this_size, dtype=self.dtype, device=self.device)
    #         for i in range(self.z_dimension):
    #             market_price[i, :] = i
    #         z[start:end] = this_pdf.mul(market_price).sum(dim=0)
    #     return z

    def predict_cdf_individual_KMMN(self, x):
        """
        Predict the cdf of market price for each instance of a given dataset x by using Kaplan–Meier estimator
        :param x: x_dimension * record_num
        :type x: torch.sparse.coo_matrix
        :return:
        """
        (_, x_size) = x.size()
        ones = torch.ones(self.z_dimension, x_size, dtype=self.dtype, device=self.device)
        s = torch.zeros(self.z_dimension, x_size, dtype=self.dtype, device=self.device)
        pdf = torch.zeros(self.z_dimension, x_size, dtype=self.dtype, device=self.device)
        cdf_complement = torch.zeros(self.z_dimension, x_size, dtype=self.dtype, device=self.device)
        omega = torch.zeros(self.z_dimension, x_size, dtype=self.dtype, device=self.device)
        omega_sum = torch.zeros(self.z_dimension, x_size, dtype=self.dtype, device=self.device)
        h = torch.zeros(self.z_dimension, x_size, dtype=self.dtype, device=self.device)

        for i in range(self.z_dimension):
            b = torch.zeros(self.b_dimension, x_size, dtype=self.dtype, device=self.device)
            b[i, :] = 1
            # b_pdf, b_omega = self.probability_z_omega_given_b_x(b, x)
            b_pdf, b_omega = self.softmax_probability_z_omega_given_b_x(b, x)
            pdf[0:i+1, :] += (b_pdf[0:i+1, :]+0) * self.b_prior[i]
            # pdf += b_pdf * self.b_prior[i]
            omega[i, :] = (b_omega+0) * self.b_prior[i]
            # omega[i, :] = b_omega+eps

        # cdf_complement[-1, :] = 0
        cdf_complement[-1, :] = pdf[-1, :]
        # omega_sum[-1, :] = 0
        omega_sum[-1, :] = omega[-1, :]
        for i in reversed(range(self.z_dimension-1)):
            # cdf_complement[i, :] = cdf_complement[i+1, :] + pdf[i+1, :]
            cdf_complement[i, :] = cdf_complement[i+1, :] + pdf[i, :]
            omega_sum[i, :] = omega_sum[i+1, :] + omega[i, :]
            # omega_sum[i, :] = omega[i, :]

        for i in range(self.z_dimension):
            # omega_sum[i, :] = omega_sum[i, :] / (self.z_dimension - i)
            # h[i, :] = torch.div(pdf[i, :], cdf_complement[i, :] + omega[i, :])
            h[i, :] = torch.div(pdf[i, :], cdf_complement[i, :] + omega_sum[i, :])

        # incase of h>1
        h[h >= ones] = 1.0

        def write_heatmap(data, tag):
            f, ax = plt.subplots(1, 1, sharex=True)
            sns.heatmap(data, ax=ax)
            self.writer.add_figure(tag, f)
            plt.close(f)
        if random.random() < 0.1:
            write_heatmap(omega_sum.sum(dim=1).view(-1, 1).cpu().numpy()/x_size, "debug/omega_sum/")
            write_heatmap(cdf_complement.sum(dim=1).view(-1, 1).cpu().numpy()/x_size, "debug/cdf_complement/")
            write_heatmap(pdf.sum(dim=1).view(-1, 1).cpu().numpy()/x_size, "debug/pdf/")
            write_heatmap(h.sum(dim=1).view(-1, 1).cpu().numpy()/x_size, "debug/h/")

        #  initialize s[0,:]
        # s[0, :] = ones[0, :] - h[0, :]
        s[0, :] = ones[0, :]
        for i in range(1, self.z_dimension):
            s[i, :] = torch.mul(s[i-1, :], (ones[i, :] - h[i, :]))
        return ones - s

    def predict_pdf_individual_KMMN(self, x):
        """
        Predict the pdf of market price for each instance of a given dataset x by using Kaplan–Meier estimator
        :param x: x_dimension * record_num
        :type x: torch.sparse.coo_matrix
        :return:
        """
        (_, x_size) = x.size()
        pdf = torch.zeros(self.z_dimension, x_size, dtype=self.dtype, device=self.device)
        cdf = self.predict_cdf_individual_KMMN(x)
        # TODO: maybe need normalization
        pdf[0, :] = torch.div(cdf[0, :] + 1e-4, cdf[-1, :])
        for i in range(1, self.z_dimension):
            # in case of zero to calculate kl-divergence and anlp
            pdf[i, :] = torch.div(cdf[i, :] - cdf[i - 1, :] + 1e-4, cdf[-1, :])
            # pdf[i, :] = cdf[i, :] - cdf[i - 1, :]
        return pdf

    def predict_z_pdf_overall_KMMN(self, x, z_origin, batch_size=8192):
        (_, x_size) = x.shape
        starts, ends = Util.generate_batch_index(x_size, batch_size)

        # cdf = torch.zeros(self.z_dimension, 1, dtype=self.dtype, device=self.device).view(-1, )
        pdf = torch.zeros(self.z_dimension, 1, dtype=self.dtype, device=self.device).view(-1, )
        ll = torch.zeros(1, 1, dtype=self.dtype, device=self.device).view(-1, )
        z = torch.zeros(x_size, 1, dtype=self.dtype, device=self.device).view(-1, )

        for start, end in list(zip(starts, ends)):
            this_size = end - start
            this_z = Util.scipy_csr_2_torch_coo(
                Util.generate_sparse(z_origin[start:end, :], np.ones((this_size,), dtype=np.int8), self.z_dimension),
                self.device, self.dtype).to_dense()
            market_price_mask = torch.zeros(self.z_dimension, this_size, dtype=self.dtype, device=self.device)
            for i in range(self.z_dimension):
                market_price_mask[i, :] = i

            this_x = Util.scipy_csr_2_torch_coo(x[:, start:end], self.device, self.dtype)
            this_pdf = self.predict_pdf_individual_KMMN(this_x)

            pdf += this_pdf.sum(dim=1).view(-1, )
            this_ll = this_z.mul(this_pdf).sum(dim=0).view(1, -1).log()
            ll += this_ll.sum()
            z[start:end] = torch.floor(this_pdf.mul(market_price_mask).sum(dim=0))

        return z, pdf / x_size, -ll[0] / x_size

    def evaluate(self, x, z_origin, b=None, omega_for_z=None, path=None, epoch=0):
        """
        Evaluate the Markov network in terms of KLD, WD and MSE
        :param x: x_dimension * record_num
        :type x: scipy.sparse.csr_matrix
        :param omega_for_z: 1 * record_num
        :type omega_for_z: numpy.ndarray
        :param z_origin: record_num * 1
        :type z_origin: numpy.ndarray
        # :param b_origin: record_num * 1
        # :type b_origin: numpy.ndarray
        :param b: b_dimension * record_num
        :type b: scipy.sparse.csr_matrix
        :param path: path to save figure
        :type path: str
        :return: KL_pdf_MN, WD_pdf_MN, KL_cdf_MN, WD_cdf_MN, MSE_MN, ANLP, omega
        """
        print("=== start evaluate ===")
        (_, x_size) = x.shape
        is_win_for_z = None
        if b is not None and omega_for_z is not None:
            MN_ANLP_z, MN_ANLP_omega = self.anlp_z_omega_given_b_x(b, x, z_origin, omega_for_z)
            self.writer.add_scalar('train/ANLP/z', MN_ANLP_z, epoch)
            self.writer.add_scalar('train/ANLP/omega', MN_ANLP_omega, epoch)
            is_win_for_z = omega_for_z < 0.5

        # ANLP, pdf_unnormalized = self.ANLP_overall(z_origin, x)
        MN_z, MN_pdf, omega, MN_ANLP = self.predict_z_pdf_omega_overall_MN(x, z_origin)
        pdf_unnormalized = 1.0
        MN_pdf = MN_pdf.cpu().numpy()
        MN_cdf = Util.pdf_to_cdf(MN_pdf)
        omega = omega.cpu().numpy()
        MN_z = MN_z.cpu().numpy()
        MN_c_index = concordance_index(z_origin, MN_z, event_observed=is_win_for_z)

        KMMN_z, KMMN_pdf, KMMN_ANLP = self.predict_z_pdf_overall_KMMN(x, z_origin)
        KMMN_ANLP = KMMN_ANLP.cpu().numpy()
        KMMN_pdf = KMMN_pdf.cpu().numpy()
        KMMN_cdf = Util.pdf_to_cdf(KMMN_pdf)
        KMMN_z = KMMN_z.cpu().numpy()
        KMMN_c_index = concordance_index(z_origin, KMMN_z, event_observed=is_win_for_z)

        pdf, cdf = Util.count_pdf_cdf(z_origin, z_lower_bound=0, z_upper_bound=self.z_dimension - 1)
        zs = list(range(self.z_dimension))

        truth_pdf = [pdf[z] if pdf[z] != 0 else 1e-6 for z in zs]
        truth_cdf = [cdf[z] if cdf[z] != 0 else 1e-6 for z in zs]

        WD_pdf_MN = wasserstein_distance(truth_pdf, MN_pdf)
        KL_pdf_MN = entropy(truth_pdf, MN_pdf)
        WD_cdf_MN = wasserstein_distance(truth_cdf, MN_cdf)
        KL_cdf_MN = entropy(truth_cdf, MN_cdf)
        MSE_MN = mean_squared_error(z_origin, MN_z)

        WD_pdf_KMMN = wasserstein_distance(truth_pdf, KMMN_pdf)
        KL_pdf_KMMN = entropy(truth_pdf, KMMN_pdf)
        WD_cdf_KMMN = wasserstein_distance(truth_cdf, KMMN_cdf)
        KL_cdf_KMMN = entropy(truth_cdf, KMMN_cdf)
        MSE_KMMN = mean_squared_error(z_origin, KMMN_z)

        print("Algorithm\tKL_pdf\tWD_pdf\tKL_cdf\tWD_cdf\tANLP\tMSE\tc_index")
        print("{7:10}\t{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}\t{4:.6f}\t{5:.6f}\t{6:.6f}"
              .format(KL_pdf_MN, WD_pdf_MN, KL_cdf_MN, WD_cdf_MN, MN_ANLP, MSE_MN, MN_c_index, "MN"))
        print("{7:10}\t{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}\t{4:.6f}\t{5:.6f}\t{6:.6f}"
              .format(KL_pdf_KMMN, WD_pdf_KMMN, KL_cdf_KMMN, WD_cdf_KMMN, KMMN_ANLP, MSE_KMMN, KMMN_c_index, "KMMN"))

        print("The quartiles of truthful market price are",
              np.percentile(np.reshape(z_origin, (-1,)), [25, 50, 75]))
        print("The quartiles of predicted market price are",
              np.percentile(np.reshape(KMMN_z, (-1,)), [25, 50, 75]))
        print("The quartiles of omega are",
              np.percentile(np.reshape(omega, (-1,)), [25, 50, 75]))
        print("The averaged omega is", omega)

        # choose a sample to draw its pdf
        index = int(np.floor(np.random.rand()*x_size))
        sample_x = Util.scipy_csr_2_torch_coo(x[:, index], self.device, self.dtype)
        sample_z = int(z_origin[index, 0])

        sample_pdf_MN, _ = self.predict_pdf_omega_individual_MN(sample_x)
        sample_pdf_MN = sample_pdf_MN.view(-1, ).cpu().numpy()
        sample_pdf_KMMN = self.predict_pdf_individual_KMMN(sample_x).view(-1, ).cpu().numpy()

        f, (ax1) = plt.subplots(1, 1, sharex=True)
        ax1.plot(zs, sample_pdf_MN, color='tab:blue', label='MN')
        ax1.plot(zs, sample_pdf_KMMN, color='tab:purple', label='KMMN')
        ax1.plot(sample_z, sample_pdf_MN[sample_z], 's-', color='r', markersize=7)
        ax1.plot(sample_z, sample_pdf_KMMN[sample_z], 'o-', color='r', markersize=7)
        ax1.axvline(linewidth=1.0, color='r', x=sample_z, linestyle='-', label=r'$z$=%d' % sample_z)

        ax1.set_ylabel("pdf")
        ax1.set_ylim(0, 0.15)
        ax1.legend()
        f.set_size_inches(7, 7 / 16 * 9)
        plt.tight_layout()
        self.writer.add_figure("sample/" + self.root_path.split("/")[-1] + "/" + str(index), f)
        plt.close(f)

        if path is not None:
            f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(zs, truth_pdf, color='tab:orange', label='truth')
            ax2.plot(zs, truth_cdf, color='tab:orange', label='truth')

            if omega_for_z is not None:
                win_pdf, win_cdf = Util.count_pdf_cdf(z_origin[omega_for_z[0, :] == 0], z_lower_bound=0,
                                                      z_upper_bound=self.z_dimension - 1)
                lose_pdf, lose_cdf = Util.count_pdf_cdf(z_origin[omega_for_z[0, :] == 1], z_lower_bound=0,
                                                        z_upper_bound=self.z_dimension - 1)
                win_pdf = [win_pdf[z] if win_pdf[z] != 0 else 1e-6 for z in zs]
                lose_pdf = [lose_pdf[z] if lose_pdf[z] != 0 else 1e-6 for z in zs]
                win_cdf = [win_cdf[z] if win_cdf[z] != 0 else 1e-6 for z in zs]
                lose_cdf = [lose_cdf[z] if lose_cdf[z] != 0 else 1e-6 for z in zs]

                ax1.plot(zs, win_pdf, color='tab:green', label='win')
                ax1.plot(zs, lose_pdf, color='tab:red', label='lose')
                ax2.plot(zs, win_cdf, color='tab:green', label='win')
                ax2.plot(zs, lose_cdf, color='tab:red', label='lose')

            ax1.plot(zs, MN_pdf, color='tab:blue', label='MN')
            ax1.plot(zs, KMMN_pdf, color='tab:purple', label='KMMN')
            ax1.set_ylabel("pdf")
            ax1.set_ylim(0, 0.10)
            ax1.legend()
            ax2.plot(zs, MN_cdf, color='tab:blue', label='MN')
            ax2.plot(zs, KMMN_cdf, color='tab:purple', label='KMMN')
            ax2.set_ylabel("cdf")
            ax2.set_ylim(0, 1)
            ax2.legend()
            f.set_size_inches(7, 7 / 16 * 9)
            plt.tight_layout()
            plt.savefig(path + ".pdf", format='pdf')
            self.writer.add_figure("evaluate/" + path.split("/")[-1], f)
            plt.close(f)
            pickle.dump(MN_pdf, open(self.root_path + '/MN_pdf', 'wb'))
            pickle.dump(KMMN_pdf, open(self.root_path + '/KMMN_pdf', 'wb'))

        return KL_pdf_KMMN, WD_pdf_KMMN, KL_cdf_KMMN, WD_cdf_KMMN, KMMN_ANLP, MSE_KMMN, omega, KMMN_c_index


if __name__ == '__main__':
    from scipy.sparse import random as srandom
    import os
    import random
    import torch
    import numpy as np

    os.environ["CUDA_VISIBLE_DEVICES"] = random.choice(['0', '1'])

    sample_size = 10000
    x_dimension = 1000
    z_dimension = 300
    b_dimension = 300

    test = Markov_Network(x_dimension=x_dimension, encoder=None, z_dimension=z_dimension, b_dimension=b_dimension,
                          device=torch.device("cuda"))

    x_origin = srandom(sample_size, x_dimension, density=0.001, format='csr')

    # b1 = srandom(b_dimension, sample_size, density=1 / b_dimension, format='csr')
    # z1 = srandom(z_dimension, sample_size, density=1 / z_dimension, format='csr')
    # b1 = np.random.rand(b_dimension, sample_size)
    # z1 = np.random.rand(z_dimension, sample_size)
    z_origin = np.random.randint(1, 300, size=(sample_size, 1))
    b_origin = np.random.randint(1, 300, size=(sample_size, 1))

    test.fit(z_origin, b_origin, x_origin)
