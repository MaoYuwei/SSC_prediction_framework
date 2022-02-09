import json
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

def plot_curve(x, y,
               m1_pred, m2_pred, c2_pred, xb_pred,
               m1_assume, m2_assume, c2_assume, xb_assume):

    x = np.array(x)
    y = np.array(y)
    b_assume = np.max(np.where(x < xb_assume))
    y_assume = [i * m1_assume for i in x[:b_assume]]
    y_assume.extend([i * m2_assume + c2_assume for i in x[b_assume:]])
    y_assume = np.array(y_assume)

    b_pred = np.max(np.where(x < xb_pred))
    y_pred = [i * m1_pred for i in x[:b_pred]]
    y_pred.extend([i * m2_pred + c2_pred for i in x[b_pred:]])
    y_pred = np.array(y_pred)

    length = 4000

    path = 'pred_autoencoder/'
    fig = plt.figure(0)

    # plt.plot(x[:length], y[:length], label='actual curve', color='b')
    # plt.plot(x[:length], y_assume[:length], label='assumed curve', color='g')
    # plt.plot(x[:length], y_pred[:length], label='predicted curve', color='r')
    #
    # # plt.title(orientation)
    # plt.legend(loc='best')
    # plt.xlabel('strain (%)')
    # plt.ylabel('stress (MPa)')
    # plt.ylim((0, 700))
    # # plt.show()
    # plt.savefig(path+orientation + '.jpg')
    # plt.close(0)

    # evaluate
    maef0 = np.mean(abs(y - y_assume) / abs(y))
    maef1 = np.mean(abs(y_assume - y_pred) / abs(y_assume))
    maef2 = np.mean(abs(y - y_pred) / abs(y))

    return maef0, maef1, maef2

# new class
class MLPRegressorOverride(MLPRegressor):
    # Overriding _init_coef method
    def _init_coef(self, fan_in, fan_out, dtype):
        factor = 6.0
        if self.activation == "logistic":
            factor = 2.0
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self._random_state.uniform(
            -init_bound, init_bound, (fan_in, fan_out)
        )
        intercept_init = self._random_state.uniform(-init_bound, init_bound, fan_out)

        # print(fan_in, fan_out)
        if fan_out==64:
            coef_init = w0
            intercept_init = w1
        if fan_in == 64:
            coef_init = w2
            intercept_init = w3

        # print('coef:', coef_init.shape)
        # print('intercept:', intercept_init.shape)

        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)

        return coef_init, intercept_init


if __name__ == '__main__':

    path = '../../auto/08/'
    nstep = 4000
    # np.random.seed(5)

    f = open('../parameters.json', 'r')
    a = json.loads(f.read())
    f.close()

    features = []
    labels = []

    orientation_list = []
    orientation_x_y = {}
    for orientation in a.keys():
        # print(orientation)

        Fp1 = np.loadtxt(path + orientation + '_Fp_tau.txt')[0]
        F1 = np.loadtxt(path + orientation + '_F_tau.txt')[0]

        first, second, _ = orientation.split('-')

        first = int(first)
        second = int(second)

        tmp = np.array([first, second])

        orientation_list.append(orientation)
        features.append(np.concatenate([Fp1, F1]))

        labels.append(a[orientation])

        x = []
        y = []
        ssy = open(path + orientation + '_py.22', 'r')
        for istep in range(nstep):
            tmp = ssy.readline().split()
            # print(tmp)
            tmpx = float(tmp[0])
            tmpy = float(tmp[1])
            x.append(tmpx)
            y.append(tmpy)

        orientation_x_y[orientation] = {'x':x, 'y':y}

    features = np.array(features)

    labels = np.array(labels)

    print(features.shape, labels.shape)

    with open('weights.json', 'r') as f:
        weighhts = json.load(f)
    w0 = np.array(weighhts['6'])
    w1 = np.array(weighhts['5'])
    w2 = np.array(weighhts['4'])
    w3 = np.array(weighhts['3'])

    print(w0.shape, w1.shape, w2.shape, w3.shape)

    hidden_layer_sizes = (64)

    maef0_list = []
    maef1_list = []
    maef2_list = []

    mse_list = []

    table0 = []
    table1 = []
    table2 = []

    for i in range(100):
        orientation = orientation_list[i]
        train_features = np.delete(features, i, axis=0)
        test_features = features[i].reshape(1, -1)
        train_labels = np.delete(labels, i, axis=0)
        test_labels = labels[i].reshape(1, -1)

        feature_scaler = StandardScaler()
        train_features = feature_scaler.fit_transform(train_features)
        label_scaler = StandardScaler()
        train_labels = label_scaler.fit_transform(train_labels)

        model = MLPRegressorOverride(hidden_layer_sizes=hidden_layer_sizes,
                             max_iter=1000,
                             activation='relu',
                             solver='lbfgs',
                             alpha=0.01,
                             random_state=1,
                             early_stopping=True)

        # print(model.coefs_)
        # print(model.intercepts_)

        model.fit(train_features, train_labels)

        test_features = feature_scaler.transform(test_features)

        pred_labels = model.predict(test_features)

        mse = mean_squared_error(y_true=label_scaler.transform(test_labels), y_pred=pred_labels)

        mse_list.append(mse)

        pred_labels = label_scaler.inverse_transform(pred_labels)
        # print(pred_labels)
        [m1_pred, m2_pred, c2_pred, xb_pred] = pred_labels[0]
        xb_pred_new = c2_pred / (m1_pred - m2_pred)

        [m1_assume, m2_assume, c2_assume, xb_assume] = test_labels[0]
        # xb_assume = c2_assume / (m1_assume - m2_assume)

        maef0, maef1, maef2 = plot_curve(orientation_x_y[orientation]['x'], orientation_x_y[orientation]['y'],
                                                  m1_pred, m2_pred, c2_pred, xb_pred_new,
                                                  m1_assume, m2_assume, c2_assume, xb_assume)

        maef0_list.append(maef0)
        maef1_list.append(maef1)
        maef2_list.append(maef2)

        print('{}, parameter mse: {}, maef0: {}, maef1: {}, maef2: {}'.format(
            orientation_list[i], mse, maef0, maef1, maef2))

        table0.append(maef0)
        table1.append(maef1)
        table2.append(maef2)

    print('parameter mse: {}, maef0: {}, maef1: {}, maef2: {}'.format(
        sum(mse_list)/len(mse_list),
        sum(maef0_list)/len(maef0_list),
        sum(maef1_list)/len(maef1_list),
        sum(maef2_list)/len(maef2_list)))

    # print('max:', max(maef2_list), maef2_list.index(max(maef2_list)))
    # print('min:', min(maef2_list), maef2_list.index(min(maef2_list)))
    #
    # tmp = sorted(maef2_list)
    # print('mean:', tmp[50], maef2_list.index(tmp[50]))
    #
    # print(orientation_list[16], orientation_list[26], orientation_list[38])
    # print(orientation_list[maef2_list.index(min(maef2_list))], orientation_list[maef2_list.index(tmp[50])],
    #       orientation_list[maef2_list.index(max(maef2_list))])

    # table0 = np.array(table0).reshape(10, 10)
    # table1 = np.array(table1).reshape(10, 10)
    # table2 = np.array(table2).reshape(10, 10)

    # ax1 = sns.heatmap(table1, linewidth=0.5)
    # plt.show()
    #
    # ax2 = sns.heatmap(table2, linewidth=0.5)
    # plt.show()

    # print(list(table1))

    # second_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    #
    # x_tick = [str(x) for x in second_list]
    # y_tick = x_tick
    #
    # pd_data0 = pd.DataFrame(table0, index=y_tick, columns=x_tick)
    # ax1 = sns.heatmap(pd_data0, linewidth=0.5, cmap='RdYlGn_r', annot=False, vmin=0, vmax=0.1).invert_yaxis()
    # plt.xlabel('Oy')
    # plt.ylabel('Ox')
    # plt.show()
    #
    # pd_data1 = pd.DataFrame(table1, index=y_tick, columns=x_tick)
    # ax1 = sns.heatmap(pd_data1, linewidth=0.5, cmap='RdYlGn_r', annot=False, vmin=0, vmax=0.1).invert_yaxis()
    # plt.xlabel('Oy')
    # plt.ylabel('Ox')
    # plt.show()
    #
    # pd_data2 = pd.DataFrame(table2, index=y_tick, columns=x_tick)
    # ax2 = sns.heatmap(pd_data2, linewidth=0.5, cmap='RdYlGn_r', annot=False, vmin=0, vmax=0.1).invert_yaxis()
    # plt.xlabel('Oy')
    # plt.ylabel('Ox')
    # plt.show()
