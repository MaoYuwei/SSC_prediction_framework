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

import pickle

def plot_curve(orientation, x, m1_pred, m2_pred, c2_pred, xb_pred):

    x = np.array(x)
    # print(xb_pred)
    b_pred = np.max(np.where(x < xb_pred))
    y_pred = [i * m1_pred for i in x[:b_pred]]
    y_pred.extend([i * m2_pred + c2_pred for i in x[b_pred:]])
    y_pred = np.array(y_pred)

    length = 4000

    path = 'pred_all/'
    fig = plt.figure(0)

    plt.plot(x[:length], y_pred[:length], label='predicted curve', color='r')

    # plt.title(orientation)
    plt.xlabel('strain (%)')
    plt.ylabel('stress (MPa)')
    plt.show()
    # plt.savefig(path+orientation + '.jpg')
    # plt.close(0)

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

    train_features = np.array(features)
    # features = np.load('Fp_F_rep.npy')

    train_labels = np.array(labels)

    print(train_features.shape, train_labels.shape)

    with open('weights.json', 'r') as f:
        weighhts = json.load(f)
    w0 = np.array(weighhts['6'])
    w1 = np.array(weighhts['5'])
    w2 = np.array(weighhts['4'])
    w3 = np.array(weighhts['3'])

    print(w0.shape, w1.shape, w2.shape, w3.shape)

    hidden_layer_sizes = (64)

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

    with open('Improved_MLP.pkl', 'wb') as fw:
        pickle.dump(model, fw)

    with open('features_scaler.pkl', 'wb') as fw:
        pickle.dump(feature_scaler, fw)

    with open('labels_scaler.pkl', 'wb') as fw:
        pickle.dump(label_scaler, fw)

    # path = '../../auto/02/ret/'
    # # for first in range(91):
    # #     for second in range(91):
    #
    # first = 0
    # second = 0
    # orientation = str(first) + '-' + str(second) + '-0'
    # # print(orientation)
    # Fp = np.loadtxt(path + orientation + '_Fp_tau.txt')[0]
    # F = np.loadtxt(path + orientation + '_F_tau.txt')[0]
    # # print(Fp, F)
    # test_features = np.concatenate([Fp, F]).reshape(1, -1)
    #
    # test_features = feature_scaler.transform(test_features)
    #
    # pred_labels = model.predict(test_features)
    #
    # pred_labels = label_scaler.inverse_transform(pred_labels)
    # # print(pred_labels)
    # [m1_pred, m2_pred, c2_pred, xb_pred] = pred_labels[0]
    # xb_pred_new = c2_pred / (m1_pred - m2_pred)
    #
    # plot_curve(orientation, orientation_x_y['0-0-0']['x'], m1_pred, m2_pred, c2_pred, xb_pred_new)





