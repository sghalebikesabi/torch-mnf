# %%
# region imports
from datetime import datetime

import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
import random
from sklearn.model_selection import train_test_split

import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

import wandb
# import sys
# sys.path.append('..')

# import os

# os.chdir("..")

from torch_mnf import models
from torch_mnf.utils import ROOT, interruptible, plot_model_preds_for_rotating_img
# endregion

# region args
parser = argparse.ArgumentParser(description='cSG-MCMC CIFAR10 Training')
parser.add_argument('--dir', type=str, default="ckpt", help='path to save checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default="data", metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--data_name', type=str, default="parkinsons")
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--regress', type=int, default=0,
                    help='if 0, classification')
parser.add_argument('--batch-size', type=int, default=16384, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--log_every', type=int, default=10,
                    help='1: SGLD; <1: SGHMC')
parser.add_argument('--lr_0', type=float, default=0.004,
                    help='lr_0')
parser.add_argument('--weight_decay', type=float, default=0.4,
                    help='lr_0')
parser.add_argument('--num_hidden', type=int, default=16,
                    help='lr_0')
parser.add_argument('--device_id',type = int, help = 'device id to use')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=100.,
                    help='temperature times dataset_size (default: 1)')

args = parser.parse_args()

batch_size = args.batch_size

plt.rc("savefig", bbox="tight", dpi=200)
plt.rcParams["figure.constrained_layout.use"] = True
plt.rc("figure", dpi=150)

device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

wandb.init(
    project="mnf",
    entity="sahra",
    reinit=True,
    settings=wandb.Settings(start_method="thread"),
    config=args,
)
# endregion
# %%
# region data
# Data
print('==> Preparing data..')
if args.regress==0 and args.data_name != "statlog":
    if args.data_name == "breast":
        data = pd.read_csv(
            "data/wdbc.data", header=None,
        )

        data[data == "?"] = np.nan
        data.dropna(axis=0, inplace=True)
        y_data = data.iloc[:, 1].values  # convert strings to integer
        x_data = data.iloc[:, 2:,].values

        # set to binary
        y_data[y_data == "B"] = 0  # benign
        y_data[y_data == "M"] = 1  # malignant
        y_data = y_data.astype("int")

    elif args.data_name == "ionosphere":
        data = pd.read_csv(
            "data/ionosphere_class.data",
            header=None,
        )
        y_data = data.iloc[:, 34].values  # convert strings to integer
        x_data = data.iloc[:, 0:34]
        x_data = x_data.drop(1, axis=1).values  # drop constant columns

        # set to binary
        y_data[y_data == "g"] = 1  # good
        y_data[y_data == "b"] = 0  # bad
        y_data = y_data.astype("int")

    elif args.data_name == "parkinsons":
        data = pd.read_csv(
            "data/parkinsons_class.data"
        )
        data[data == "?"] = np.nan
        data.dropna(axis=0, inplace=True)
        y_data = data["status"].values  # convert strings to integer
        x_data = data.drop(columns=["name", "status"]).values

    elif args.data_name == "mnist":
        # y, lab = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        # k = ckern.Conv(GPflow.kernels.RBF(25, ARD=self.run_settings['kernel_ard']), [28, 28],
        #                [5, 5]) + GPflow.kernels.White(1, 1e-3)
        from torchvision.datasets import MNIST

        train_data = MNIST(args.data_path, train=True, download=True)
        x_data = train_data.data.numpy().reshape((-1, 784))
        y_data = train_data.targets.numpy()
        y_data, x_data = y_data[y_data < 2], x_data[y_data < 2]
        
        trn_idx, val_idx = train_test_split(
            np.arange(len(y_data)), test_size=0.05, random_state=args.seed + 3
        )
        y_val = np.concatenate((x_data[val_idx], y_data[val_idx][:, None]), axis=1)

        x_data = x_data[trn_idx]
        y_data = y_data[trn_idx]
        test_data = MNIST(args.data_path, train=False, download=True)
        x_test = test_data.data.numpy().reshape((-1, 784))
        y_test = test_data.targets.numpy()
        y_test = np.concatenate(
            (x_test[y_test < 2], y_test[y_test < 2][:, None]), axis=1
        )
        y = np.concatenate((x_data, y_data[:, None]), axis=1)
        to_drop = np.where(np.std(y, axis=0) == 0)[0]
        y = np.delete(y, to_drop, 1)
        y_val = np.delete(y_val, to_drop, 1)
        y_test = np.delete(y_test, to_drop, 1)

        norm_idx = y.shape[1] - 1
        y[:, :norm_idx] = y[:, :norm_idx] + np.random.uniform(
            0, 1, size=y[:, :norm_idx].shape
        )
        y_val[:, :norm_idx] = y_val[:, :norm_idx] + np.random.uniform(
            0, 1, size=y_val[:, :norm_idx].shape
        )
        y_test[:, :norm_idx] = y_test[:, :norm_idx] + np.random.uniform(
            0, 1, size=y_test[:, :norm_idx].shape
        )
        y = y.astype("float")
        y_val = y_val.astype("float")
        y_test = y_test.astype("float")
        y[:, :norm_idx] = y[:, :norm_idx] / 256.0
        y_val[:, :norm_idx] = y_val[:, :norm_idx] / 256.0
        y_test[:, :norm_idx] = y_test[:, :norm_idx] / 256.0
        
        def logit(x):
            return np.log(x / (1.0 - x))

        def logit_transform(x):
            return logit(1e-10 + (1 - 2e-10) * x)

        y[:, :norm_idx] = logit_transform(y[:, :norm_idx])
        y_val[:, :norm_idx] = logit_transform(y_val[:, :norm_idx])
        y_test[:, :norm_idx] = logit_transform(y_test[:, :norm_idx])

        x_train = y[:, :norm_idx]
        y_train = y[:, norm_idx].astype("int")
        x_test = y_test[:, :norm_idx]
        y_test = y_test[:, norm_idx].astype("int")

    else:
        print("Dataset doesn't exist")
        raise NotImplementedError


elif args.regress:
    if args.data_name == "concrete":
        data = pd.read_excel(
            "data/Concrete_Data.xls"
        )
        y_data = data.iloc[:, 8].values
        x_data = data.iloc[:, 0:8].values
    elif args.data_name == "wine":
        data = pd.read_csv(
            "data/winequality-red.csv", sep=";"
        )
        y_data = data.iloc[:, 11].values  # convert strings to integer
        x_data = data.iloc[:, 0:11].values
    elif args.data_name == "boston":
        from sklearn.datasets import load_boston

        x_data, y_data = load_boston(return_X_y=True)
    elif args.data_name == "diabetes":
        from sklearn.datasets import load_diabetes

        x_data, y_data = load_diabetes(return_X_y=True)
    else:
        print("Dataset doesn't exist")
        raise NotImplementedError
    
if args.data_name != 'mnist':
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.5, random_state=args.seed
    )

# scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
args.test_batch_size = min(args.batch_size, x_test.shape[0])
args.batch_size = min(args.batch_size, x_train.shape[0])

if args.regress:
    scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train = scaler.transform(y_train.reshape(-1, 1))
    y_test = scaler.transform(y_test.reshape(-1, 1))

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float() if args.regress else torch.from_numpy(y_train)),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_train).float() if args.regress else torch.from_numpy(y_test)),
    batch_size=args.test_batch_size,
    shuffle=False,
    num_workers=0,
)
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_train).float() if args.regress else torch.from_numpy(y_test)
# endregion

# %%
def train_step(model, optim, loss_fn, images, labels):
    # We could draw multiple posterior samples here to get unbiased Monte Carlo
    # estimate for the NLL which would decrease training variance but slow us down.
    optim.zero_grad()
    preds = model(images)
    loss = loss_fn(preds, labels)
    loss.backward()
    optim.step()
    return loss, preds


@interruptible
def train_fn(model, optim, loss_fn, data_loader, epochs=1, log_every=30, writer=None):
    vars(model).setdefault("step", 0)  # add train step counter on model if none exists

    for epoch in range(epochs):
        pbar = tqdm(data_loader, desc=f"epoch {epoch + 1}/{epochs}")
        for samples, labels in pbar:
            model.step += 1

            loss, preds = train_step(model, optim, loss_fn, samples, labels)

            if log_every and model.step % log_every == 0:
                # Accuracy estimated by single call for speed. Would be more
                # accurate to approximately integrate over parameter posteriors
                # by averaging across multiple calls.
                val_preds = model(x_test)
                if args.regress == 0:
                    val_acc = (y_test == val_preds.argmax(1)).float().mean()
                    train_acc = (labels == preds.argmax(1)).float().mean()
                    pbar.set_postfix(loss=f"{loss:.3}", val_acc=f"{val_acc:.3}")

                    wandb.log({"accuracy/training": train_acc}, step=model.step)
                    wandb.log({"accuracy/validation": val_acc}, step=model.step)

                    if writer:
                        writer.add_scalar("accuracy/training", train_acc, model.step)
                        writer.add_scalar("accuracy/validation", val_acc, model.step)
                        
                val_loss = loss_fn(val_preds, y_test)
                wandb.log({"loss/training": loss}, step=model.step)
                wandb.log({"loss/validation": val_loss}, step=model.step)


# %%
mnf_lenet = models.MNFFeedForward(layer_sizes = [x_train.shape[1], args.num_hidden, 2])
mnf_adam = torch.optim.Adam(mnf_lenet.parameters(), lr=args.lr_0)
print(f"MNFLeNet param count: {sum(p.numel() for p in mnf_lenet.parameters()):,}")

writer = None

if args.regress:
    criterion = nn.GaussianNLLLoss()
else:
    criterion = F.nll_loss

# %%
def mnf_loss_fn(preds, labels):
    if args.regress:
        nll = criterion(preds[:, 0], labels, preds[:, 1].exp()).mean()
    else:
        nll = criterion(preds, labels).mean()

    # The KL divergence acts as a regularizer to prevent overfitting.
    kl_div = mnf_lenet.kl_div() / len(train_loader)
    loss = nll + kl_div

    # writer.add_scalar("loss/NLL", nll, mnf_lenet.step)
    # writer.add_scalar("loss/KL div", kl_div, mnf_lenet.step)
    # writer.add_scalar("loss/NLL + KL", loss, mnf_lenet.step)

    return loss

# %%
train_fn(mnf_lenet, mnf_adam, mnf_loss_fn, train_loader, writer=writer, epochs=args.epochs, log_every=args.log_every)

val_preds = mnf_lenet(x_test)
if args.regress == 0:
    def predict_density(inputs, all_preds):
        inputs = inputs.float()
        logp = torch.log(torch.nn.functional.softmax(all_preds, dim=1))
        return inputs * logp[:, 1] + (1 - inputs) * logp[:, 0]
    nll = -predict_density(y_test, val_preds).mean()

else:
    nll = criterion(val_preds[:, 0], y_test, val_preds[:, 1].exp()).mean()
    
print('nll', nll.item())

wandb.log({'nll': nll.item()})