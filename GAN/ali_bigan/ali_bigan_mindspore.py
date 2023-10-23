import numpy as np
import mindspore
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm
from mindspore import Parameter, Tensor
from mindspore.dataset import vision
from mindspore.dataset import MnistDataset
from download import download
from itertools import *


mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="GPU")

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "../../mnist", kind="zip", replace=True)

train_dataset = MnistDataset("../../mnist/MNIST_Data/train", shuffle=False)

mb_size = 128
train_dataset = train_dataset.batch(batch_size=mb_size)
train_dataset = train_dataset.map(vision.Rescale(1.0 / 255.0, 0), input_columns='image')
depth, on_value, off_value = Tensor(10, mindspore.int32), Tensor(1.0, mindspore.float32), Tensor(0.0, mindspore.float32)
# y_train = mindspore.ops.one_hot(Tensor(y_train, mindspore.int32), depth, on_value, off_value, axis=-1)
X_dim = 28*28
y_dim = 10
z_dim = 10
h_dim = 128
cnt = 0
lr = 1e-3

def log(x):
    return mindspore.ops.log(x + 1e-8)


# Inference net (Encoder) Q(z|X)
Q = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, z_dim)
)

# Generator net (Decoder) P(X|z)
P = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(z_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)

D_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim + z_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, 1),
    mindspore.nn.Sigmoid()
)


def D(X, z):
    return D_(mindspore.ops.cat([X, z], axis=1))

def D_loss_fn(X, z):
    z_hat = Q(X)
    X_hat = P(z)

    D_enc = D(X, z_hat)
    D_gen = D(X_hat, z)
    D_loss = -mindspore.ops.mean(log(D_enc) + log(1 - D_gen))
    return D_loss

def G_loss_fn(D_gen, D_enc):
    G_loss = -mindspore.ops.mean(log(D_gen) + log(1 - D_enc))
    return G_loss

Q.update_parameters_name('Q')
P.update_parameters_name('P')
D_.update_parameters_name('D_')

D_param_len = len(D_.trainable_params())
Q_param_len = len(Q.trainable_params())
P_param_len = len(P.trainable_params())
weights = [*Q.trainable_params(), *P.trainable_params(), *D_.trainable_params()]
D_grad_fn = mindspore.value_and_grad(D_loss_fn, None, weights)
weights = [*Q.trainable_params(), *P.trainable_params()]
G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, weights)

G_solver = mindspore.nn.Adam(chain(Q.trainable_params(), P.trainable_params()), learning_rate=lr)
D_solver = mindspore.nn.Adam(D_.trainable_params(), learning_rate=lr)

data_loader = train_dataset.create_tuple_iterator()

for it in tqdm(range(1000000)):
    # Sample data
    try:
        X, _ = next(data_loader)
    except StopIteration:
        data_loader = train_dataset.create_tuple_iterator()
        continue
    z = Parameter(mindspore.ops.randn(X.shape[0], z_dim))
    X = X.flatten().view(X.shape[0], -1)
    X = Parameter(X)
    # Discriminator
    D_loss, D_grad = D_grad_fn(X, z)
    D_solver(D_grad[-D_param_len:])
    G_solver(D_grad[:-D_param_len])

    # Autoencoder Q, P
    z_hat = Q(X)
    X_hat = P(z)

    D_enc = D(X, z_hat)
    D_gen = D(X_hat, z)

    G_loss, G_grad = G_grad_fn(D_gen, D_enc)
    G_solver(G_grad)

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.numpy(), G_loss.numpy()))

        samples = P(z).numpy()[:16]

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

        if not os.path.exists('out/'):
            os.makedirs('out/')

        plt.savefig('out/{}.png'.format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
