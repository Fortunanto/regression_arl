import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
import regressor
import torch.nn.functional as F
from arl import ARL
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.nc import QuantileRegErrFunc


import numpy as np
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use your desired seed here
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', type=int, default=6)
args = parser.parse_args()
seed_everything(args.seed)
def sample_from_2d_gaussian(mean, cov,num_samples=1):
    mv_normal = MultivariateNormal(mean, cov)
    sample = mv_normal.sample((num_samples,))
    return sample
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def create_icp(n_train,x_train,y_train):     
    # divide the data into proper training set and calibration set
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]
    in_shape = x_train.shape[1]
    alpha = 0.1

    # zero mean and unit variance scaling 
    scalerX = StandardScaler()
    scalerX = scalerX.fit(x_train[idx_train])

    # scale
    x_train = scalerX.transform(x_train)

    # scale the labels by dividing each by the mean absolute response
    mean_y_train = np.mean(np.abs(y_train[idx_train]))
    y_train = np.squeeze(y_train)/mean_y_train
    #########################################################
    # Quantile random forests parameters
    # (See QuantileForestRegressorAdapter class in helper.py)
    #########################################################

    # when tuning the two QRF quantile levels one may
    # ask for a prediction band with smaller average coverage
    # to avoid too conservative estimation of the prediction band
    # This would be equal to coverage_factor*(quantiles[1] - quantiles[0])
    # coverage_factor = 0.85

    # ratio of held-out data, used in cross-validation
    cv_test_ratio = 0.05

    # seed for splitting the data in cross-validation.
    # Also used as the seed in quantile random forests function
    cv_random_state = 1

    # determines the lowest and highest quantile level parameters.
    # This is used when tuning the quanitle levels by cross-validation.
    # The smallest value is equal to quantiles[0] - range_vals.
    # Similarly, the largest value is equal to quantiles[1] + range_vals.
    # cv_range_vals = 30

    # sweep over a grid of length num_vals when tuning QRF's quantile parameters                   
    # cv_num_vals = 10

    # pytorch's optimizer object
    nn_learn_func = torch.optim.Adam

    # number of epochs
    epochs = 100

    # learning rate
    lr = 0.0005

    # mini-batch size
    batch_size = 64

    # hidden dimension of the network
    hidden_size = 32

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
    # print(icp)
    return icp

# Example usage
centroids = torch.tensor([(1,1),(1,1),(1,1)]).float()
cov = torch.tensor([[[2.0, 1], [1, 2.0]],[[2.0, 1], [1, 2.0]],[[100,10],[10,30]]])
n_samples = [10000,10000,200]
samples = []
target = []
for i,centroid in enumerate(centroids):
    num_samples = n_samples[i]
    samples_cur = sample_from_2d_gaussian(centroid, cov[i], num_samples)
    samples = samples + [samples_cur]
    target += [centroid.unsqueeze(0)]*num_samples
total_targets = torch.cat(target, dim=0)
total_samples = torch.cat(samples, dim=0)
from sklearn.model_selection import train_test_split

# Concatenate the targets and samples into one tensor
data = torch.cat((total_samples, total_targets), dim=1)

# Convert to numpy for splitting
data_np = data.numpy()

# Perform an 80-10 split first to separate the training set
train_val_data, test_data = train_test_split(data_np, test_size=0.1, random_state=42)

# Then split the training set further into training and validation sets (80-10-10 split overall)
train_data, val_data = train_test_split(train_val_data, test_size=1/9, random_state=42)

# Convert back to torch.Tensor
train_data = torch.tensor(train_data, dtype=torch.float32)
val_data = torch.tensor(val_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)
# Separate the features (samples) from the targets
X_train, y_train = train_data[:, :2], train_data[:, 2:]
X_val, y_val = val_data[:, :2], val_data[:, 2:]
X_test, y_test = test_data[:, :2], test_data[:, 2:]
# icp = create_icp(X_train.shape[0],X_train.cpu().numpy(),y_train.cpu().numpy())

# Now you can convert your data to PyTorch DataLoader, which makes it easier to handle mini-batches

from torch.utils.data import TensorDataset, DataLoader

batch_size = 256  # Adjust based on your needs

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model and optimizer
model = ARL()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer model to correct device.
model = model.to(device)

# Adagrad is the defeault optimizer.
optimizer_learner = torch.optim.Adagrad(
    model.learner.parameters(), lr=1e-1
)
optimizer_adv = torch.optim.Adagrad(
    model.adversary.parameters(), lr=1e-1
)

batch_losses = []
epoch_losses = []
val_epoch_losses = []
total_steps = 0
from tqdm import tqdm
for epoch in tqdm(range(30)):
    for step, (train_point, train_target) in enumerate(train_loader):
        # Transfer data to GPU if possible. 
        train_point = train_point.to(device)
        # train_point = train_point / train_p
        train_target = train_target.to(device)
        total_steps += 1
        out = model.learner_step(train_point)
        adv_weights = model.adversary_step(out).squeeze()
        loss_learner = (((out-train_target)**2).sum(dim=-1)*adv_weights).sum()
        optimizer_learner.zero_grad()
        optimizer_adv.zero_grad()
        # freeze_model(model.adversary)
        # unfreeze_model(model.learner)
        loss_learner.backward(retain_graph=True)
        optimizer_learner.step()
        # if epoch>=1:
        loss_adv = -(((out-train_target)**2).sum(dim=-1)*adv_weights).sum()
        # freeze_model(model.learner)
        # unfreeze_model(model.adversary)
        loss_adv.backward()
        optimizer_adv.step()
            # if epoch>=2:
            # print(loss_learner)
            # Learner update step.
            
        # batch_losses.append(loss.item())

    # Average loss for this epoch
    # avg_loss = sum(batch_losses[-len(train_loader):]) / len(train_loader)
    # epoch_losses.append(avg_loss)

    # # Calculate validation loss
    # with torch.no_grad():
    #     val_losses = [F.mse_loss(regressor(X_val), y_val) for X_val, y_val in val_loader]
    #     avg_val_loss = sum(val_losses) / len(val_losses)
    #     val_epoch_losses.append(avg_val_loss)

# Calculate test loss

out = []
errors = []
dict_loss = {}

with torch.no_grad():
    for batch in test_loader:
        X_test, y_test = batch
        X_test, y_test = X_test.to(device="cuda"), y_test.to(device="cuda")
        # dict_loss[y_test.cpu().numpy()] = []
        yhat = model.learner(X_test)
        error = torch.sqrt(((yhat-y_test)**2).sum(dim=-1))
        for i,centroid in enumerate(y_test):
            centroid = centroid.cpu().numpy()
            if (centroid[0],centroid[1]) not in dict_loss:
                dict_loss[(centroid[0],centroid[1])] = []
            dict_loss[(centroid[0],centroid[1])]+= [error[i].item()]
        out += [yhat]
        errors += [error]
    out = torch.cat(out, dim=0).cpu().numpy()
    errors = torch.cat(errors, dim=0).cpu().numpy()  # collect errors
    for key in dict_loss:
        dict_loss[key] = np.mean(dict_loss[key])
    
    print(f"{errors.mean()}",end='')
    x = out[:,0]
    y = out[:,1]
    plt.scatter(x, y, c=errors, cmap='viridis')  # use errors for color
    plt.colorbar(label='Error')
    plt.savefig("test_arl_no_cqr.png")

    # test_losses = [F.mse_loss(regressor(X_test.to(device="cuda")), y_test.to(device="cuda")).cpu() for X_test, y_test in test_loader]
    # avg_test_loss = sum(test_losses) / len(test_losses)

# print("Average test loss: ", avg_test_loss)

# # Plot the batch losses
# plt.figure(figsize=(18, 6))
# plt.subplot(1, 3, 1)
# plt.plot(batch_losses)
# plt.title("Batch Loss over All Batches")
# plt.xlabel("Batch")
# plt.ylabel("Loss")

# # Plot the average epoch losses
# plt.subplot(1, 3, 2)
# plt.plot(epoch_losses)
# plt.title("Average Training Loss over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Average Loss")

# # Plot the average validation losses
# plt.subplot(1, 3, 3)
# plt.plot(val_epoch_losses)
# plt.title("Average Validation Loss over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Average Loss")

# plt.tight_layout()
# plt.savefig("losses.png")
