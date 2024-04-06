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
z_dim = 16
X_dim = 28*28
h_dim = 128
cnt = 0
lr = 1e-3
lam1 = 1e-2
lam2 = 1e-2

def log(x):
    return mindspore.ops.log(x + 1e-8)


E = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, z_dim)
)

G = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(z_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)

D = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, 1),
    mindspore.nn.Sigmoid()
)

data_loader = train_dataset.create_tuple_iterator()

def sample_X(size, include_y=False):
    global data_loader
    X = None
    while X is None:
        try:
            X, y = next(data_loader)
            if X.shape[0] != mb_size:
                X = None
                continue
        except StopIteration:
            data_loader = train_dataset.create_tuple_iterator()
            continue

    X = X.flatten().view(X.shape[0], -1)
    X = Parameter(X, requires_grad=False)

    if include_y:
        y = np.argmax(y.numpy(), axis=1).astype(np.int)
        y = Parameter(mindspore.Tensor.from_numpy(y), requires_grad=False)
        return X, y

    return X


G_loss_fn = lambda D_fake, reg: -mindspore.ops.mean(log(D_fake)) + reg
D_loss_fn = lambda D_real, D_fake: -mindspore.ops.mean(log(D_real) + log(1 - D_fake))
E_loss_fn = lambda mse, D_reg: mindspore.ops.mean(lam1 * mse + lam2 * log(D_reg))
G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G.trainable_params())
D_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D.trainable_params())
E_grad_fn = mindspore.value_and_grad(E_loss_fn, None, E.trainable_params())


E_solver = mindspore.nn.Adam(E.trainable_params(), learning_rate=lr)
G_solver = mindspore.nn.Adam(G.trainable_params(), learning_rate=lr)
D_solver = mindspore.nn.Adam(D.trainable_params(), learning_rate=lr)

for it in range(1000000):
    """ Discriminator """
    # Sample data
    X = sample_X(mb_size)
    z = Parameter(mindspore.ops.randn(mb_size, z_dim), requires_grad=False)

    # Dicriminator_1 forward-loss-backward-update
    G_sample = G(z)
    D_real = D(X)
    D_fake = D(G_sample)

    D_loss, D_grad = D_grad_fn(D_real, D_fake)
    D_solver(D_grad)

    """ Generator """
    # Sample data
    X = sample_X(mb_size)
    z = Parameter(mindspore.ops.randn(mb_size, z_dim), requires_grad=False)

    # Generator forward-loss-backward-update
    G_sample = G(z)
    G_sample_reg = G(E(X))
    D_fake = D(G_sample)
    D_reg = D(G_sample_reg)

    mse = mindspore.ops.sum((X - G_sample_reg)**2, 1)
    reg = mindspore.ops.mean(lam1 * mse + lam2 * log(D_reg))
    G_loss, G_grad = G_grad_fn(D_fake, reg)
    G_solver(G_grad)


    """ Encoder """
    # Sample data
    X = sample_X(mb_size)
    z = Parameter(mindspore.ops.randn(mb_size, z_dim), requires_grad=False)

    G_sample_reg = G(E(X))
    D_reg = D(G_sample_reg)

    mse = mindspore.ops.sum((X - G_sample_reg)**2, 1)
    E_loss, E_grad = E_grad_fn(mse, D_reg)
    E_solver(E_grad)

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {}; E_loss: {}; G_loss: {}'
              .format(it, D_loss.numpy(), E_loss.numpy(), G_loss.numpy()))

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

        plt.savefig('out/{}.png'
                    .format(str(cnt).zfill(3)), bbox_inches='tight')
        cnt += 1
        plt.close(fig)
