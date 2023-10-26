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
d_step = 3
lr = 1e-3
m = 5


G = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(z_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)
G.update_parameters_name('G')

# D is an autoencoder
D_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
)
D_.update_parameters_name('D_')

# Energy is the MSE of autoencoder
def D(X):
    X_recon = D_(X)
    return mindspore.ops.mean(mindspore.ops.sum((X - X_recon)**2, 1))


G_solver = mindspore.nn.Adam(G.trainable_params(), learning_rate=lr)
G_loss_fn = lambda D_fake: D_fake
G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G.trainable_params())
D_solver = mindspore.nn.Adam(D_.trainable_params(), learning_rate=lr)
D_loss_fn = lambda D_real, D_fake: D_real + mindspore.ops.relu(m - D_fake)
D_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D_.trainable_params())


data_loader = train_dataset.create_tuple_iterator()

for it in tqdm(range(1000000)):
    # Sample data
    z = mindspore.ops.randn(mb_size, z_dim)
    try:
        X, _ = next(data_loader)
    except StopIteration:
        data_loader = train_dataset.create_tuple_iterator()
        continue
    if X.shape[0] != mb_size:
        continue

    X = X.flatten().view(X.shape[0], -1)
    # Dicriminator
    G_sample = G(z)
    D_real = D(X)
    D_fake = D(G_sample)

    # Reuse D_fake for generator loss
    D_loss, D_grad = D_grad_fn(D_real, D_fake)
    D_solver(D_grad)

    # Generator
    G_sample = G(z)
    D_fake = D(G_sample)

    G_loss, G_grad = G_grad_fn(D_fake)
    G_solver(G_grad)

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.numpy().item(), G_loss.numpy().item()))

        samples = G(z).numpy()[:16]

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
