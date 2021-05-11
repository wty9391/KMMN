import numpy as np
from scipy.stats import entropy
import random
import pickle

from util import util

SAVE_MEMORY_MODE = False
maximal_train_record = 3000000

class NodeInfo:
    nodeIndex = 0
    bestFeat = 0
    KLD = 0.0
    s1_keys = []    # str
    s2_keys = []    # str
    def __init__(self,_nodeIndex,_bestFeat,_KLD,_s1_keys,_s2_keys):
        self.nodeIndex = _nodeIndex
        self.bestFeat = _bestFeat
        self.KLD = _KLD
        self.s1_keys = _s1_keys
        self.s2_keys = _s2_keys

class KMDT:
    def __init__(self, IFROOT, result_root, OFROOT, max_market_price=301):
        self.IFROOT = IFROOT
        self.result_root = result_root
        self.OFROOT = OFROOT
        self.BASE_BID = '0'

        self.UPPER = max_market_price
        self.LAPLACE = 3
        self.NORMAL = 0
        self.SURVIVAL = 1
        self.FULL = 2  # only use in baseline
        self.EVAL_MODE_LIST = ['0']
        self.MODE_NAME_LIST = ['normal', 'survival', 'full']

        self.LEAF_SIZE = 3000
        self.TREE_DEPTH = 10
        self.MODE_LIST = [self.SURVIVAL]

        # self.FEATURE_LIST = [9, 10, 11, 16]
        self.FEATURE_LIST = [1, 2, 9, 10, 11, 16, 17, 18, 21]
        self.FEAT_NAME = ['click', 'weekday', 'hour', 'bidid', 'timestamp',
                          'logtype', 'ipinyouid', 'useragent', 'IP', 'region',
                          'city', 'adexchange', 'domain', 'url', 'urlid',
                          'slotid', 'slotwidth', 'slotheight', 'slotvisibility', 'slotformat',
                          'slotprice', 'creative', 'bidprice', 'payprice', 'keypage',
                          'advertiser', 'usertag', 'index', 'mybidprice', 'winAuction']
        self.PAY_PRICE_INDEX = self.FEAT_NAME.index('payprice')

        if self.result_root.find("criteo") >= 0:
            print("criteo initialize")
            self.FEATURE_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.FEAT_NAME = ['campaign', 'cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8',
                              'cost', 'index', 'mybidprice', 'winAuction']
            self.PAY_PRICE_INDEX = self.FEAT_NAME.index('cost')

        self.MY_BID_INDEX = self.FEAT_NAME.index('mybidprice')
        self.WIN_AUCTION_INDEX = self.FEAT_NAME.index('winAuction')
        # self.PAY_PRICE_INDEX = self.FEAT_NAME.index('z')

        self.nodeData = {}
        self.nodeInfos = {}

    def getTrainData(self):
        campgain = self.result_root.split("/")[-1]
        f = open(self.IFROOT + "/" + campgain + "/train.log.txt", 'r')
        z_train = pickle.load(open(self.result_root + '/z_train', 'rb'))
        b_train = pickle.load(open(self.result_root + '/b_train', 'rb'))
        is_win = b_train > z_train

        dataset = []
        f.seek(0)
        f.readline()  # first line is header
        i = 0
        for line in f:
            if SAVE_MEMORY_MODE and i >= maximal_train_record:
                continue
            items = line.split('\t')
            items.append(str(i))  # index from 0
            items.append(str(int(b_train[i][0])))
            items.append(str(int(is_win[i][0])))
            # items.append(str(int(z_train[i][0])))
            dataset.append(items)
            i += 1
        f.close()
        return dataset

    def getTestData(self):
        campgain = self.result_root.split("/")[-1]
        f = open(self.IFROOT + "/" + campgain + "/test.log.txt", 'r')
        # z_test = pickle.load(open(self.result_root + '/z_test', 'rb'))

        dataset = []
        f.seek(0)
        f.readline()  # first line is header
        i = 0
        for line in f:
            items = line.split('\t')
            items.append(str(i))  # ith row from i=0
            # items.append(0)
            # items.append(0)
            # items.append(str(int(z_test[i][0])))
            dataset.append(items)
            i += 1
        f.close()
        return dataset

    def dataset2s(self, dataset, featIndex):
        s = {}  # {featValue: [count]}
        winbids = {}  # {featValue: {z:count}}
        losebids = {}  # {featValue: {b:count}}
        for i in range(len(dataset)):
            pay_price = int(dataset[i][self.PAY_PRICE_INDEX])
            mybidprice = int(dataset[i][self.MY_BID_INDEX])
            win = int(dataset[i][self.WIN_AUCTION_INDEX])
            featValue = dataset[i][featIndex]
            if featValue not in losebids:
                losebids[featValue] = {}
            if featValue not in winbids:
                winbids[featValue] = {}

            if win == 0:
                if mybidprice not in losebids[featValue]:
                    losebids[featValue][mybidprice] = 0
                losebids[featValue][mybidprice] += 1
                continue
            if win == 1:
                if pay_price not in winbids[featValue]:
                    winbids[featValue][pay_price] = 0
                winbids[featValue][pay_price] += 1

                if featValue not in s:
                    s[featValue] = [0]*self.UPPER
                s[featValue][pay_price] += 1

        return s, winbids, losebids

    def calProbDistribution(self, winbids, losebids):
        """

        :param winbids: {nodeIndex: {z: count}}
        :param losebids: {nodeIndex: {b: count}}
        :return: pdf: []
        """

        counts_z = [0] * self.UPPER
        counts_b = [0] * self.UPPER

        for nodeIndex, z_count in winbids.items():
            for z, count in z_count.items():
                counts_z[z] += count

        for nodeIndex, b_count in losebids.items():
            for b, count in b_count.items():
                counts_b[b] += count

        win_z_count = []  # win cases with z=i
        lose_b_sum_count = []  # lose cases with b>=i
        win_z_sum_count = []  # win cases with z>=i

        for i in range(self.UPPER):
            win_z_count.append(counts_z[i])
            lose_b_sum_count.append(counts_b[i])

        lose_b_sum_count = [sum(lose_b_sum_count[i:]) for i in range(self.UPPER)]
        win_z_sum_count = [sum(win_z_count[i:]) for i in range(self.UPPER)]

        hs = []
        ws = []
        for i in range(self.UPPER):
            d = win_z_count[i]
            n = lose_b_sum_count[i] + win_z_sum_count[i]
            h = 1 - d / n if n > 0 else 1
            hs.append(h)

        for i in range(self.UPPER):
            s = np.prod(hs[:i + 1])
            w = 1 - s
            ws.append(w)

        return util.Util.cdf_to_pdf(ws)

    def s2dataset(self, s, orgDataset, featIndex):
        dataset = []
        for i in range(len(orgDataset)):
            if orgDataset[i][featIndex] in s:
                dataset.append(orgDataset[i])
        return dataset

    def kmeans(self, s, winbids, losebids):
        if len(s) == 0:
            return 0

        leafSize = self.LEAF_SIZE
        s1 = {}
        s2 = {}
        winbids1 = {}
        winbids2 = {}
        losebids1 = {}
        losebids2 = {}
        len1 = 0
        len2 = 0
        lenk = {}
        # random split s into s1 and s2, and calculate minPrice,maxPrice
        for i in range(int(len(s) / 2)):
            k = list(s.keys())[i]
            s1[k] = s[k]
            winbids1[k] = winbids[k]
            losebids1[k] = losebids[k]
            lenk[k] = sum(s[k])
            len1 += lenk[k]
        for i in range(int(len(s) / 2), len(s)):
            k = list(s.keys())[i]
            s2[k] = s[k]
            winbids2[k] = winbids[k]
            losebids2[k] = losebids[k]
            lenk[k] = sum(s[k])
            len2 += lenk[k]
        # EM-step
        KLD1 = 0.0
        KLD2 = 0.0
        KLD = 0.0
        pr = []
        count = 0
        isBreak = 0
        not_converged = 1
        while not_converged:
            count += 1
            # begin
            not_converged = 0
            # E-step:
            q1 = self.calProbDistribution(winbids1, losebids1)
            q2 = self.calProbDistribution(winbids2, losebids2)
            KLD = entropy(q1, q2)
            if count > 8 and KLD < KLD1:
                isBreak = 1
            if count > 3 and KLD < KLD1 and KLD == KLD2:
                isBreak = 1
            KLD2 = KLD1
            KLD1 = KLD
            # M-step:
            for k in s.keys():
                mk = self.calProbDistribution({k: winbids[k]}, {k: losebids[k]})
                k1 = entropy(mk, q1)
                k2 = entropy(mk, q2)
                if k1 < k2:
                    if k in s1:
                        continue
                    if len2 - lenk[k] < leafSize:
                        continue
                    not_converged = 1
                    s1[k] = s[k]
                    winbids1[k] = winbids[k]
                    losebids1[k] = losebids[k]
                    len1 += lenk[k]
                    if k in s2:
                        len2 -= lenk[k]
                        s2.pop(k)
                        winbids2.pop(k)
                        losebids2.pop(k)
                elif k1 > k2:
                    if k in s2:
                        continue
                    if len1 - lenk[k] < leafSize:
                        continue
                    not_converged = 1
                    s2[k] = s[k]
                    winbids2[k] = winbids[k]
                    losebids2[k] = losebids[k]
                    len2 += lenk[k]
                    if k in s1:
                        len1 -= lenk[k]
                        s1.pop(k)
                        winbids1.pop(k)
                        losebids1.pop(k)
            if isBreak == 1:
                break
        return s1, s2, winbids1, winbids2, losebids1, losebids2

    def train(self):
        dataset = self.getTrainData()
        # priceSet = [int(data[self.PAY_PRICE_INDEX]) for data in dataset]

        iStack = []
        iStack.append(1)
        dataStack = []
        dataStack.append(dataset.copy())
        while len(iStack) != 0:
            nodeIndex = iStack.pop()
            dataset = dataStack.pop()
            print("nodeIndex = " + str(nodeIndex))
            if 2 * nodeIndex >= 2 ** self.TREE_DEPTH:
                self.nodeData[nodeIndex] = dataset.copy()
                continue

            maxKLD = -1.0
            bestFeat = 0
            count = 0  # detect if there's no feature to split
            for featIndex in range(len(dataset[0])):
                if featIndex not in self.FEATURE_LIST:
                    continue
                s, winbids, losebids = self.dataset2s(dataset, featIndex)
                if len(s.keys()) <= 1:
                    continue
                count += 1
                tmpS1, tmpS2, winbids1, winbids2, losebids1, losebids2 = self.kmeans(s, winbids, losebids)
                q1 = self.calProbDistribution(winbids1, losebids1)
                q2 = self.calProbDistribution(winbids2, losebids2)
                KLD = entropy(q1, q2)
                if count == 1:
                    maxKLD = KLD
                    bestFeat = featIndex
                    s1 = tmpS1.copy()
                    s2 = tmpS2.copy()
                if maxKLD < KLD and len(tmpS1) != 0 and len(tmpS2) != 0:
                    maxKLD = KLD
                    bestFeat = featIndex
                    s1 = tmpS1.copy()
                    s2 = tmpS2.copy()

            if count == 0 or len(s1.keys()) == 0 or len(s2.keys()) == 0:  # no feature can split
                self.nodeData[nodeIndex] = dataset.copy()
                continue
            dataset1 = self.s2dataset(s1, dataset, bestFeat)
            dataset2 = self.s2dataset(s2, dataset, bestFeat)

            if len(dataset1) < self.LEAF_SIZE or len(dataset2) < self.LEAF_SIZE:
                self.nodeData[nodeIndex] = dataset.copy()
                continue

            self.nodeInfos[nodeIndex] = NodeInfo(
                nodeIndex, bestFeat, maxKLD, list(s1.keys()).copy(), list(s2.keys()).copy())

            if len(dataset2) > 2 * self.LEAF_SIZE:
                iStack.append(2 * nodeIndex + 1)
                dataStack.append(dataset2.copy())
            else:
                self.nodeData[2 * nodeIndex + 1] = dataset2.copy()
            if len(dataset1) > 2 * self.LEAF_SIZE:
                iStack.append(2 * nodeIndex)
                dataStack.append(dataset1.copy())
            else:
                self.nodeData[2 * nodeIndex] = dataset1.copy()

        return

    def getTrainPriceCount(self):
        wcount = {}
        winbids = {}  # {nodeIndex: {z: count}}
        losebids = {}  # {nodeIndex: {b: count}}
        mp = {}  # full {nodeIndex: {z: count}}

        for nodeIndex, values in self.nodeData.items():
            if nodeIndex not in wcount:
                wcount[nodeIndex] = [0] * self.UPPER
                winbids[nodeIndex] = {}
                losebids[nodeIndex] = {}
                mp[nodeIndex] = [0] * self.UPPER

            for v in values:
                if len(v) < self.WIN_AUCTION_INDEX:
                    print("item length error")
                    continue

                pay_price = int(v[self.PAY_PRICE_INDEX])
                mp[nodeIndex][pay_price] += 1
                wcount[nodeIndex][pay_price] += 1

                if int(v[self.WIN_AUCTION_INDEX]) == 0:  # lose
                    mybidprice = int(v[self.MY_BID_INDEX])
                    if mybidprice not in losebids[nodeIndex]:
                        losebids[nodeIndex][mybidprice] = 0
                    losebids[nodeIndex][mybidprice] += 1

                if int(v[self.WIN_AUCTION_INDEX]) == 1:  # win
                    if pay_price not in winbids[nodeIndex]:
                        winbids[nodeIndex][pay_price] = 0
                    winbids[nodeIndex][pay_price] += 1

        return wcount, winbids, losebids, mp

    def q2w(self, q):
        w = 0.
        if isinstance(q, dict):
            w = {}
            for k in q.keys():
                w[k] = [0.] * len(q[k])
                for i in range(1, len(q[k])):
                    w[k][i] += w[k][i - 1] + q[k][i - 1]
        if isinstance(q, list):
            w = [0.] * len(q)
            for i in range(1, len(q)):
                w[i] += w[i - 1] + q[i - 1]
        return w

    def getQ(self):
        q = {}  # {nodeIndex: {z: count}}
        w = {}
        num = {}  # num of
        bnum = {}
        wcount, winbids, losebids, _ = self.getTrainPriceCount()

        # get q
        for k in wcount:
            q[k] = self.calProbDistribution({k: winbids[k]}, {k: losebids[k]})
            w[k] = self.q2w(q[k])
            num[k] = sum(wcount[k])
            b1 = set(winbids[k].keys())
            b2 = set(losebids[k].keys())
            b = b1 | b2
            bnum[k] = len(b)

        return q

    def getN(self):
        testset = self.getTestData()
        n = {}  # {nodeIndex: [z_count]}
        priceSet = [int(data[self.PAY_PRICE_INDEX]) for data in testset]

        for i in range(len(testset)):
            nodeIndex = 1
            pay_price = int(testset[i][self.PAY_PRICE_INDEX])
            if len(self.nodeInfos.keys()) == 0:
                if nodeIndex not in n:
                    n[nodeIndex] = [0.] * self.UPPER
                n[nodeIndex][pay_price] += 1
                continue

            while True:
                bestFeat = self.nodeInfos[nodeIndex].bestFeat
                s1_keys = self.nodeInfos[nodeIndex].s1_keys
                s2_keys = self.nodeInfos[nodeIndex].s2_keys
                if testset[i][bestFeat] in s1_keys:
                    nodeIndex = 2 * nodeIndex
                elif testset[i][bestFeat] in s2_keys:
                    nodeIndex = 2 * nodeIndex + 1
                else:  # feature value doesn't appear in train data
                    nodeIndex = 2 * nodeIndex + random.randint(0, 1)

                if nodeIndex not in self.nodeInfos:
                    if nodeIndex not in n:
                        n[nodeIndex] = [0.] * self.UPPER
                    n[nodeIndex][pay_price] += 1
                    break

        return n

    def evaluate(self, x, z):
        """
        x and z are not used
        :param x:
        :param z:
        :return:
        """
        (record_size, _) = np.shape(x)
        q = self.getQ()  # for train, q contains pdf of each node
        n = self.getN()  # for test, n contains z-count number of each node

        node_predict_z = {}
        node_predict_count = {}
        predict_pdf = np.zeros((self.UPPER, ))
        total_count = 0
        mse = 0.0
        anlp = 0.0

        for nodeIndex, pdf in q.items():
            # nomalize pdf
            pdf_sum = sum(pdf)
            q[nodeIndex] = [v/pdf_sum for v in pdf]

            node_predict_z[nodeIndex] = sum([i * pdf[i] for i in range(len(pdf))])

        for nodeIndex, count in n.items():
            node_predict_count[nodeIndex] = sum(count)
            total_count += sum(count)

        assert total_count == record_size, "output size must be equal to input size"

        # q.nodeIndex maybe not in n.nodeIndex
        for nodeIndex, _ in n.items():
            mse += sum([n[nodeIndex][truth_z] *
                        (truth_z-node_predict_z[nodeIndex])**2 for truth_z in range(len(n[nodeIndex]))])

        for nodeIndex, count in n.items():
            anlp += sum([-np.log(q[nodeIndex][truth_z]) * count[truth_z] for truth_z in range(len(count))])

        for nodeIndex, count in n.items():
            predict_pdf += np.array(q[nodeIndex]) * sum(count)

        mse = mse / total_count
        anlp = anlp / total_count
        predict_pdf = predict_pdf / total_count

        return mse, anlp, predict_pdf.tolist()

