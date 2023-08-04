import torch
import random
import numpy as np
# np.warnings.filterwarnings('ignore')

# from datasets import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.nc import QuantileRegErrFunc

seed = 1

random_state_train_test = seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
# desired miscoverage error
alpha = 0.1

# desired quanitile levels
quantiles = [0.05, 0.95]

# used to determine the size of test set
test_ratio = 0.2

# name of dataset
dataset_base_path = "./datasets/"
dataset_name = "community"

# load the dataset
X, y = datasets.GetDataset(dataset_name, dataset_base_path)

# divide the dataset into test and train based on the test_ratio parameter
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=test_ratio,
                                                    random_state=random_state_train_test)

# reshape the data
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

# compute input dimensions
n_train = x_train.shape[0]
in_shape = x_train.shape[1]

# display basic information
print("Dataset: %s" % (dataset_name))
print("Dimensions: train set (n=%d, p=%d) ; test set (n=%d, p=%d)" % 
      (x_train.shape[0], x_train.shape[1], x_test.shape[0], x_test.shape[1]))

# divide the data into proper training set and calibration set
idx = np.random.permutation(n_train)
n_half = int(np.floor(n_train/2))
idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]

# zero mean and unit variance scaling 
scalerX = StandardScaler()
scalerX = scalerX.fit(x_train[idx_train])

# scale
x_train = scalerX.transform(x_train)
x_test = scalerX.transform(x_test)

# scale the labels by dividing each by the mean absolute response
mean_y_train = np.mean(np.abs(y_train[idx_train]))
y_train = np.squeeze(y_train)/mean_y_train
y_test = np.squeeze(y_test)/mean_y_train
#########################################################
# Quantile random forests parameters
# (See QuantileForestRegressorAdapter class in helper.py)
#########################################################

# when tuning the two QRF quantile levels one may
# ask for a prediction band with smaller average coverage
# to avoid too conservative estimation of the prediction band
# This would be equal to coverage_factor*(quantiles[1] - quantiles[0])
coverage_factor = 0.85

# ratio of held-out data, used in cross-validation
cv_test_ratio = 0.05

# seed for splitting the data in cross-validation.
# Also used as the seed in quantile random forests function
cv_random_state = 1

# determines the lowest and highest quantile level parameters.
# This is used when tuning the quanitle levels by cross-validation.
# The smallest value is equal to quantiles[0] - range_vals.
# Similarly, the largest value is equal to quantiles[1] + range_vals.
cv_range_vals = 30

# sweep over a grid of length num_vals when tuning QRF's quantile parameters                   
cv_num_vals = 10

# pytorch's optimizer object
nn_learn_func = torch.optim.Adam

# number of epochs
epochs = 1000

# learning rate
lr = 0.0005

# mini-batch size
batch_size = 64

# hidden dimension of the network
hidden_size = 64

# dropout regularization rate
dropout = 0.1

# weight decay regularization
wd = 1e-6

# Ask for a reduced coverage when tuning the network parameters by 
# cross-validataion to avoid too concervative initial estimation of the 
# prediction interval. This estimation will be conformalized by CQR.
quantiles_net = [0.1, 0.9]
# define quantile neural network model
quantile_estimator = helper.AllQNet_RegressorAdapter(model=None,
                                                     fit_params=None,
                                                     in_shape=in_shape,
                                                     hidden_size=hidden_size,
                                                     quantiles=quantiles_net,
                                                     learn_func=nn_learn_func,
                                                     epochs=epochs,
                                                     batch_size=batch_size,
                                                     dropout=dropout,
                                                     lr=lr,
                                                     wd=wd,
                                                     test_ratio=cv_test_ratio,
                                                     random_state=cv_random_state,
                                                     use_rearrangement=False)

# define a CQR object, computes the absolute residual error of points 
# located outside the estimated quantile neural network band 
nc = RegressorNc(quantile_estimator, QuantileRegErrFunc())

# run CQR procedure
icp = helper.train_icp(nc, x_train, y_train, idx_train, idx_cal, alpha)
prediction = icp.predict(x_train,alpha)
y_lower=prediction[:,0]
y_upper=prediction[:,1]
print(y_lower,y_upper)
