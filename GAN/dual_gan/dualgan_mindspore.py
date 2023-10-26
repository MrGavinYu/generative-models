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
z_dim = 10
X_dim = 28*28
y_dim = 10
h_dim = 128
cnt = 0
lr = 1e-4
n_critics = 3
lam1, lam2 = 100, 100


def log(x):
    return mindspore.ops.log(x + 1e-8)


G1 = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim + z_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)
G1.update_parameters_name('G1')

G2 = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim + z_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)
G2.update_parameters_name('G2')

D1 = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, 1)
)
D1.update_parameters_name('D1')

D2 = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, 1)
)
D2.update_parameters_name('D2')

G_solver = mindspore.nn.RMSProp(chain(G1.trainable_params(), G2.trainable_params()), learning_rate=lr)
G_loss_fn = lambda G_loss, reg1, reg: G_loss + reg1 + reg
G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, chain(G1.trainable_params(), G2.trainable_params()))
D1_solver = mindspore.nn.RMSProp(D1.trainable_params(), learning_rate=lr)
D1_loss_fn = lambda D1_real, D1_fake: -(mindspore.ops.mean(D1_real) - mindspore.ops.mean(D1_fake))
D1_grad_fn = mindspore.value_and_grad(D1_loss_fn, None, D1.trainable_params())
D2_solver = mindspore.nn.RMSProp(D2.trainable_params(), learning_rate=lr)
D2_loss_fn = lambda D2_real, D2_fake: -(mindspore.ops.mean(D2_real) - mindspore.ops.mean(D2_fake))
D2_grad_fn = mindspore.value_and_grad(D2_loss_fn, None, D2.trainable_params())

dataset_len = train_dataset.get_dataset_size()
half = int(dataset_len / 2)

# Real image
X_train1, X_train2 = train_dataset.split([half, dataset_len-half])
# Rotated image
X_train2 = X_train2.map(vision.Rotate(90), input_columns='image')

X_train1 = X_train1.shuffle(100)
X_train2 = X_train2.shuffle(100)

data_loader_X1 = X_train1.create_tuple_iterator()
data_loader_X2 = X_train2.create_tuple_iterator()

for it in tqdm(range(1000000)):
    for _ in range(n_critics):
        # Sample data
        z1 = mindspore.ops.randn(mb_size, z_dim)
        z2 = mindspore.ops.randn(mb_size, z_dim)
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

        X1 = X1.flatten().view(X1.shape[0], -1)
        X2 = X2.flatten().view(X2.shape[0], -1)
        # D1
        X2_sample = G1(mindspore.ops.cat([X1, z1], 1))  # G1: X1 -> X2
        D1_real = D1(X2)
        D1_fake = D1(X2_sample)

        D1_loss, D1_grad = D1_grad_fn(D1_real, D1_fake)
        D1_solver(D1_grad)

        # Weight clipping
        for p in D1.trainable_params():
            p.set_data(p.data.clamp(-0.01, 0.01))

        # D2
        X1_sample = G2(mindspore.ops.cat([X2, z2], 1))  # G2: X2 -> X1
        D2_real = D2(X1)
        D2_fake = D2(X1_sample)

        D2_loss, D2_grad = D2_grad_fn(D2_real, D2_fake)
        D2_solver(D2_grad)

        # Weight clipping
        for p in D2.trainable_params():
            p.set_data(p.data.clamp(-0.01, 0.01))

    # Generator
    z1 = mindspore.ops.randn(mb_size, z_dim)
    z2 = mindspore.ops.randn(mb_size, z_dim)
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

    X1 = X1.flatten().view(X1.shape[0], -1)
    X2 = X2.flatten().view(X2.shape[0], -1)
    X1_sample = G2(mindspore.ops.cat([X2, z2], 1))
    X2_sample = G1(mindspore.ops.cat([X1, z1], 1))

    X1_recon = G2(mindspore.ops.cat([X2_sample, z2], 1))
    X2_recon = G1(mindspore.ops.cat([X1_sample, z1], 1))

    D1_fake = D1(X1_sample)
    D2_fake = D2(X2_sample)

    G_loss = -mindspore.ops.mean(D1_fake) - mindspore.ops.mean(D2_fake)
    reg1 = lam1 * mindspore.ops.mean(mindspore.ops.sum(mindspore.ops.abs(X1_recon - X1), 1))
    reg2 = lam2 * mindspore.ops.mean(mindspore.ops.sum(mindspore.ops.abs(X2_recon - X2), 1))

    G_loss, G_grad = G_grad_fn(G_loss, reg1, reg2)
    G_solver(G_grad)

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D1_loss.numpy().item() + D2_loss.numpy().item(), G_loss.numpy().item()))

        real1 = X1.numpy()[:4]
        real2 = X2.numpy()[:4]
        samples1 = X1_sample.numpy()[:4]
        samples2 = X2_sample.numpy()[:4]
        samples = np.vstack([real2, samples1, real1, samples2])

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
