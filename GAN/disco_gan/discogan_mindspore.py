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
lr = 1e-3


def log(x):
    return mindspore.ops.log(x + 1e-8)


def plot(samples):
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

    return fig


G_AB = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)
G_AB.update_parameters_name('G_AB')

G_BA = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)
G_BA.update_parameters_name('G_BA')

D_A = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, 1),
    mindspore.nn.Sigmoid()
)
D_A.update_parameters_name('D_A')

D_B = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, 1),
    mindspore.nn.Sigmoid()
)
D_B.update_parameters_name('D_B')

nets = [G_AB, G_BA, D_A, D_B]
G_params = list(G_AB.trainable_params()) + list(G_BA.trainable_params())
D_params = list(D_A.trainable_params()) + list(D_B.trainable_params())


G_solver = mindspore.nn.Adam(G_params, learning_rate=lr)
G_loss_fn = lambda L_G_AB, L_G_BA: L_G_AB + L_G_BA
G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G_params)
D_solver = mindspore.nn.Adam(D_params, learning_rate=lr)
D_loss_fn = lambda L_D_A, L_D_B: L_D_A + L_D_B
D_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D_params)


if not os.path.exists('out/'):
    os.makedirs('out/')

# Gather training data: domain1 <- real MNIST img, domain2 <- rotated MNIST img
# train_dataset.dataset_size == 60000
dataset_len = train_dataset.get_dataset_size()
half = int(dataset_len / 2)

# Real image
X_train1, X_train2 = train_dataset.split([half, dataset_len-half])
# Rotated image
X_train2 = X_train2.map(vision.Rotate(90), input_columns='image')

data_loader_X1 = X_train1.create_tuple_iterator()
data_loader_X2 = X_train2.create_tuple_iterator()

# Training
for it in tqdm(range(1000000)):
    # Sample data from both domains
    try:
        X_A, _ = next(data_loader_X1)
    except StopIteration:
        X_train1 = X_train1.shuffle(100)
        data_loader_X1 = X_train1.create_tuple_iterator()
        continue
    try:
        X_B, _ = next(data_loader_X2)
    except StopIteration:
        X_train2 = X_train2.shuffle(100)
        data_loader_X2 = X_train2.create_tuple_iterator()
        continue
    if X_A.shape[0] != mb_size or X_B.shape[0] != mb_size:
        continue
    z = mindspore.ops.randn(mb_size, z_dim)
    X_A = X_A.flatten().view(X_A.shape[0], -1)
    X_B = X_B.flatten().view(X_B.shape[0], -1)

    # Discriminator A
    X_BA = G_BA(X_B)
    D_A_real = D_A(X_A)
    D_A_fake = D_A(X_BA)

    L_D_A = -mindspore.ops.mean(log(D_A_real) + log(1 - D_A_fake))

    # Discriminator B
    X_AB = G_AB(X_A)
    D_B_real = D_B(X_B)
    D_B_fake = D_B(X_AB)

    L_D_B = -mindspore.ops.mean(log(D_B_real) + log(1 - D_B_fake))

    # Total discriminator loss
    D_loss, D_grad = D_grad_fn(L_D_A, L_D_B)
    D_solver(D_grad)

    # Generator AB
    X_AB = G_AB(X_A)
    D_B_fake = D_B(X_AB)
    X_ABA = G_BA(X_AB)

    L_adv_B = -mindspore.ops.mean(log(D_B_fake))
    L_recon_A = mindspore.ops.mean(mindspore.ops.sum((X_A - X_ABA)**2, 1))
    L_G_AB = L_adv_B + L_recon_A

    # Generator BA
    X_BA = G_BA(X_B)
    D_A_fake = D_A(X_BA)
    X_BAB = G_AB(X_BA)

    L_adv_A = -mindspore.ops.mean(log(D_A_fake))
    L_recon_B = mindspore.ops.mean(mindspore.ops.sum((X_B - X_BAB)**2, 1))
    L_G_BA = L_adv_A + L_recon_B

    # Total generator loss
    G_loss, G_grad = G_grad_fn(L_G_AB, L_G_BA)
    G_solver(G_grad)

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.numpy().item(), G_loss.numpy().item()))
        try:
            input_A, _ = next(data_loader_X1)
            input_B, _ = next(data_loader_X2)
        except StopIteration:
            continue
        if input_A.shape[0] < 4 or input_B.shape[0] < 4:
            continue
        input_A = input_A.flatten().view(input_A.shape[0], -1)[:4, :]
        input_B = input_B.flatten().view(input_B.shape[0], -1)[:4, :]

        samples_A = G_BA(input_B).numpy()
        samples_B = G_AB(input_A).numpy()

        input_A = input_A.numpy()
        input_B = input_B.numpy()

        # The resulting image sample would be in 4 rows:
        # row 1: real data from domain A, row 2 is its domain B translation
        # row 3: real data from domain B, row 4 is its domain A translation
        samples = np.vstack([input_A, samples_B, input_B, samples_A])

        fig = plot(samples)
        plt.savefig('out/{}.png'
                    .format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
