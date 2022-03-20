# Python script for Efficiency prediction with SVR method, given input power(1D)
# Look into: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
import numpy as np
from sklearn.svm import SVR
from scipy import io
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt


def split_data():
    mat = io.loadmat('SVR.mat')
    Eff = mat['Eff']
    Pin = mat['Pin']
    Pin_train, Pin_test, Eff_train, Eff_test = train_test_split(
        Pin, Eff, test_size=0.3
    )
    print(Pin_train.shape)
    print(Pin_test.shape)
    print(Eff_train.shape)
    print(Eff_test.shape)
    return Pin_train, Pin_test, Eff_train, Eff_test


Pin_train, Pin_test, Eff_train, Eff_test = split_data()

# param = {'C':np.linspace(10, 1000, 100), 'gamma':np.linspace(0.00001, 1, 100)}
# #  C is the parameter of error tolerance, C is higher while error is lower
# #  gamma is the parameter of gaussian function
# svr_rbf = SVR(kernel='rbf', cache_size=1000)
# grid_search = GridSearchCV(svr_rbf, param, n_jobs=8, verbose=1)
# grid_search.fit(Pin_train, Eff_train.ravel())
# best_param = grid_search.best_estimator_.get_params()
# for para, val in list(best_param.items()):
#     print(para, val)


svr_rbf_best = SVR(kernel='rbf', C=10, gamma=0.01011)
svr_rbf_best.fit(Pin_train, Eff_train.ravel())
y_rbf = svr_rbf_best.predict(Pin_test)

lw = 2
plt.scatter(Pin_test, Eff_test, color='darkorange', label='data')
plt.scatter(Pin_test, y_rbf, color='navy', lw=lw, label='RBF model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()