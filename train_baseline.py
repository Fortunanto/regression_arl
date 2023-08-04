import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import pyplot as plt
import regressor
import torch.nn.functional as F
from arl import ARL

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
model = regressor.Regressor(2,2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer model to correct device.
model = model.to(device)

# Adagrad is the defeault optimizer.
optimizer_learner = torch.optim.Adagrad(
    model.parameters(), lr=1e-1
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
        train_target = train_target.to(device)
        total_steps += 1
        out = model(train_point)
        # adv_weights = model.adversary_step(out).squeeze()
        loss_learner = F.mse_loss(out, train_target)
        optimizer_learner.zero_grad()
        # optimizer_adv.zero_grad()
        # freeze_model(model.adversary)
        # unfreeze_model(model.learner)
        loss_learner.backward()
        optimizer_learner.step()
        # if epoch>=1:
        # loss_adv = -(((out-loss_learner)**2).sum(dim=-1)*adv_weights).sum()
        # freeze_model(model.learner)
        # unfreeze_model(model.adversary)
        # loss_adv.backward()
        # optimizer_adv.step()
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
        yhat = model(X_test)
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
    # print(dict_loss)
    print(f"{errors.mean()}",end='')
    x = out[:,0]
    y = out[:,1]
    plt.scatter(x, y, c=errors, cmap='viridis')  # use errors for color
    plt.colorbar(label='Error')
    plt.savefig("test_no_arl_no_cqr.png")

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
