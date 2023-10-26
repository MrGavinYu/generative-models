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
# train_dataset = train_dataset.batch(batch_size=mb_size)
train_dataset = train_dataset.map(vision.Rescale(1.0 / 255.0, 0), input_columns='image')
z_dim = 100
X_dim = 28*28
y_dim = 10
h_dim = 128
cnt = 0
lr = 1e-3


""" Shared Generator weights """
G_shared = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(z_dim, h_dim),
    mindspore.nn.ReLU(),
)
G_shared.update_parameters_name('G_shared')

""" Generator 1 """
G1_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)
G1_.update_parameters_name('G1_')

""" Generator 2 """
G2_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)
G2_.update_parameters_name('G2_')


def G1(z):
    h = G_shared(z)
    X = G1_(h)
    return X


def G2(z):
    h = G_shared(z)
    X = G2_(h)
    return X


""" Shared Discriminator weights """
D_shared = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(h_dim, 1),
    mindspore.nn.Sigmoid()
)
D_shared.update_parameters_name('D_shared')

""" Discriminator 1 """
D1_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU()
)
D1_.update_parameters_name('D1_')

""" Discriminator 2 """
D2_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU()
)
D2_.update_parameters_name('D2_')

def D1(X):
    h = D1_(X)
    y = D_shared(h)
    return y


def D2(X):
    h = D2_(X)
    y = D_shared(h)
    return y


D_params = (list(D1_.trainable_params()) + list(D2_.trainable_params()) +
            list(D_shared.trainable_params()))
G_params = (list(G1_.trainable_params()) + list(G2_.trainable_params()) +
            list(G_shared.trainable_params()))
nets = [G_shared, G1_, G2_, D_shared, D1_, D2_]

G_solver = mindspore.nn.Adam(G_params, learning_rate=lr)
G_loss_fn = lambda G1_loss, G2_loss: G1_loss + G2_loss
G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G_params)
D_solver = mindspore.nn.Adam(D_params, learning_rate=lr)
D_loss_fn = lambda D1_loss, D2_loss: D1_loss + D2_loss
D_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D_params)

# train_dataset.dataset_size == 60000
dataset_len = train_dataset.get_dataset_size()
half = int(dataset_len / 2)

# Real image
X_train1, X_train2 = train_dataset.split([half, dataset_len-half])
# Rotated image
X_train2 = X_train2.map(vision.Rotate(90), input_columns='image')

def sample_x(X, size):
    start_idx = np.random.randint(0, X.shape[0]-size)
    return X[start_idx:start_idx+size]

X_train1 = X_train1.shuffle(100).batch(batch_size=mb_size)
X_train2 = X_train2.shuffle(100).batch(batch_size=mb_size)

data_loader_X1 = X_train1.create_tuple_iterator()
data_loader_X2 = X_train2.create_tuple_iterator()

for it in tqdm(range(100000)):
    try:
        X1, _ = next(data_loader_X1)
    except StopIteration:
        X_train1 = X_train1.shuffle(100)
        data_loader_X1 = X_train1.create_tuple_iterator()
        continue
    try:
        X2, _ = next(data_loader_X2)
    except StopIteration:
        X_train2 = X_train2.shuffle(100)
        data_loader_X2 = X_train2.create_tuple_iterator()
        continue
    if X1.shape[0] != mb_size or X2.shape[0] != mb_size:
        continue
    z = mindspore.ops.randn(mb_size, z_dim)
    X1 = X1.flatten().view(X1.shape[0], -1)
    X2 = X2.flatten().view(X2.shape[0], -1)
    # Dicriminator
    G1_sample = G1(z)
    D1_real = D1(X1)
    D1_fake = D1(G1_sample)

    G2_sample = G2(z)
    D2_real = D2(X2)
    D2_fake = D2(G2_sample)

    D1_loss = mindspore.ops.mean(-mindspore.ops.log(D1_real + 1e-8) -
                         mindspore.ops.log(1. - D1_fake + 1e-8))
    D2_loss = mindspore.ops.mean(-mindspore.ops.log(D2_real + 1e-8) -
                         mindspore.ops.log(1. - D2_fake + 1e-8))
    D_loss, D_grad = D_grad_fn(D1_loss, D2_loss)

    # Average the gradients
    # for p in D_shared.parameters():
    #     p.grad.data = 0.5 * p.grad.data
    D_grad = list(D_grad)
    for i in range(len(D_shared.trainable_params())):
        D_grad[-i] = D_grad[-i] * 0.5
    D_solver(tuple(D_grad))

    # Generator
    G1_sample = G1(z)
    D1_fake = D1(G1_sample)

    G2_sample = G2(z)
    D2_fake = D2(G2_sample)

    G1_loss = mindspore.ops.mean(-mindspore.ops.log(D1_fake + 1e-8))
    G2_loss = mindspore.ops.mean(-mindspore.ops.log(D2_fake + 1e-8))
    G_loss, G_grad = G_grad_fn(G1_loss, G2_loss)

    # Average the gradients
    # for p in G_shared.parameters():
    #     p.grad.data = 0.5 * p.grad.data
    G_grad = list(G_grad)
    for i in range(len(G_shared.trainable_params())):
        G_grad[-i] = G_grad[-i] * 0.5
    G_solver(tuple(G_grad))

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D1_loss: {:.4}; G1_loss: {:.4}; '
              'D2_loss: {:.4}; G2_loss: {:.4}'
              .format(
                  it, D1_loss.numpy().item(), G1_loss.numpy().item(),
                  D2_loss.numpy().item(), G2_loss.numpy().item())
              )

        z = mindspore.ops.randn(8, z_dim)
        samples1 = G1(z).numpy()
        samples2 = G2(z).numpy()
        samples = np.vstack([samples1, samples2])

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

        plt.savefig('out/{}.png'
                    .format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
