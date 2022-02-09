
import json
import numpy as np
import pickle

from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
import warnings
import pandas as pd
import CP_PowerLaw
warnings.filterwarnings('ignore')

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
        #
        if fan_in == 18 and fan_out==64:
            coef_init = w0
            intercept_init = w1
        if fan_in == 64 and fan_out==4:
            coef_init = w2
            intercept_init = w3

        # print('coef:', coef_init.shape)
        # print('intercept:', intercept_init.shape)

        coef_init = coef_init.astype(dtype, copy=False)
        intercept_init = intercept_init.astype(dtype, copy=False)

        return coef_init, intercept_init

x = []
ssy = open('0-0-0_py.22', 'r')
for istep in range(4000):
    tmp = ssy.readline().split()
    tmpx = float(tmp[0])
    x.append(tmpx)
x = np.array(x)

def do_predict(ox, oy):

    orientation = str(ox)+'-'+str(oy)+'-0'

    # predict initial Fp and F
    f1_list = [ox, oy, 0]
    CP_PowerLaw.main(f1_list)

    # predict curve
    Fp = np.loadtxt('Fp_tau.txt')[0]
    F = np.loadtxt('F_tau.txt')[0]
    # print(Fp, F)
    test_features = np.concatenate([Fp, F]).reshape(1, -1)

    # print(test_features)

    model = pickle.load(open('Improved_MLP.pkl', 'rb'))
    feature_scaler = pickle.load(open('features_scaler.pkl', 'rb'))
    label_scaler = pickle.load(open('labels_scaler.pkl', 'rb'))

    test_features = feature_scaler.transform(test_features)

    pred_labels = model.predict(test_features)

    pred_labels = label_scaler.inverse_transform(pred_labels)

    [m1_pred, m2_pred, c2_pred, xb_pred] = pred_labels[0]
    xb_pred_new = c2_pred / (m1_pred - m2_pred)

    b_pred = np.max(np.where(x < xb_pred_new))
    y_pred = [i * m1_pred for i in x[:b_pred]]
    y_pred.extend([i * m2_pred + c2_pred for i in x[b_pred:]])
    y_pred = np.array(y_pred)

    length = 4000

    fig = plt.figure(0)

    plt.title(orientation)
    plt.plot(x[:length], y_pred[:length], label='predicted curve', color='r')
    plt.xlabel('strain (%)')
    plt.ylabel('stress (MPa)')

    plt.show()
    plt.savefig('image.jpg')
    plt.close(0)



if __name__ == '__main__':
    with open('weights.json', 'r') as f:
        weighhts = json.load(f)
    w0 = np.array(weighhts['6'])
    w1 = np.array(weighhts['5'])
    w2 = np.array(weighhts['4'])
    w3 = np.array(weighhts['3'])

    do_predict(ox=0, oy=0)
