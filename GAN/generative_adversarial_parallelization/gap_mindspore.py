import mindspore
import mindspore.nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import random
from tqdm import tqdm
from mindspore import Parameter, Tensor
from mindspore.dataset import vision
from mindspore.dataset import MnistDataset
from download import download


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
z_dim = 10
h_dim = 128
cnt = 0
lr = 1e-3
K = 100


def log(x):
    return mindspore.ops.log(x + 1e-8)


G1_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(z_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)


D1_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, 1),
    mindspore.nn.Sigmoid()
)

G2_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(z_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)


D2_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, 1),
    mindspore.nn.Sigmoid()
)

nets = [G1_, D1_, G2_, D2_]


D_loss_fn = lambda D_real, D_fake: -mindspore.ops.mean(log(D_real) + log(1 - D_fake))
G_loss_fn = lambda D_fake: -mindspore.ops.mean(log(D_fake))

G1_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G1_.trainable_params())
G2_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G2_.trainable_params())
D1_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D1_.trainable_params())
D2_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D2_.trainable_params())

G1_solver = mindspore.nn.Adam(G1_.trainable_params(), learning_rate=lr)
D1_solver = mindspore.nn.Adam(D1_.trainable_params(), learning_rate=lr)
G2_solver = mindspore.nn.Adam(G2_.trainable_params(), learning_rate=lr)
D2_solver = mindspore.nn.Adam(D2_.trainable_params(), learning_rate=lr)

D1 = {'model': D1_, 'solver': D1_solver, 'grad': D1_grad_fn}
G1 = {'model': G1_, 'solver': G1_solver, 'grad': G1_grad_fn}
D2 = {'model': D2_, 'solver': D2_solver, 'grad': D2_grad_fn}
G2 = {'model': G2_, 'solver': G2_solver, 'grad': G2_grad_fn}

data_loader = train_dataset.create_tuple_iterator()
GAN_pairs = [(D1, G1), (D2, G2)]

for it in range(1000000):
    # Sample data
    try:
        X, _ = next(data_loader)
    except StopIteration:
        data_loader = train_dataset.create_tuple_iterator()
        continue
    X = X.flatten().view(X.shape[0], -1)
    X = Parameter(X, requires_grad=False)
    z = Parameter(mindspore.ops.randn(X.shape[0], z_dim), requires_grad=False)

    for D, G in GAN_pairs:
        # Discriminator
        G_sample = G['model'](z)
        D_real = D['model'](X)
        D_fake = D['model'](G_sample)

        D_loss, D_grad = D['grad'](D_real, D_fake)

        D['solver'](D_grad)

        # Generator
        G_sample = G['model'](z)
        D_fake = D['model'](G_sample)

        G_loss, G_grad = G['grad'](D_fake)
        G['solver'](G_grad)

    if it != 0 and it % K == 0:
        # Swap (D, G) pairs
        new_D1, new_D2 = GAN_pairs[1][0], GAN_pairs[0][0]
        GAN_pairs = [(new_D1, G1), (new_D2, G2)]

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.numpy(), G_loss.numpy()))

        # Pick G randomly
        G_rand = random.choice([G1_, G2_])
        samples = G_rand(z).numpy()[:16]

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
