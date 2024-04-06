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
Z_dim = 16
X_dim = 28*28
h_dim = 128
cnt = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Parameter(mindspore.ops.randn(*size) * xavier_stddev, requires_grad=True)


""" ==================== GENERATOR ======================== """

Wzh = xavier_init(size=[Z_dim + 10, h_dim])
Wzh.name = 'Wzh'
bzh = Parameter(mindspore.ops.zeros(h_dim), requires_grad=True)
bzh.name = 'bzh'

Whx = xavier_init(size=[h_dim, X_dim])
Whx.name = 'Whx'
bhx = Parameter(mindspore.ops.zeros(X_dim), requires_grad=True)
bhx.name = 'bhx'


def G(z, c):
    inputs = mindspore.ops.cat([z, c], 1)
    h = mindspore.ops.relu(inputs @ Wzh + bzh.reshape(-1, 1).repeat(inputs.shape[0], 1))
    X = mindspore.ops.sigmoid(h @ Whx + bhx.reshape(1, -1).repeat(h.shape[0], 0))
    return X


""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[X_dim, h_dim])
Wxh.name = 'Wxh'
bxh = Parameter(mindspore.ops.zeros(h_dim), requires_grad=True)
bxh.name = 'bxh'

Why = xavier_init(size=[h_dim, 1])
Why.name = 'Why'
bhy = Parameter(mindspore.ops.zeros(1), requires_grad=True)
bhy.name = 'bhy'


def D(X):
    h = mindspore.ops.relu(X @ Wxh + bxh.reshape(-1, 1).repeat(X.shape[0], 1))
    y = mindspore.ops.sigmoid(h @ Why + bhy.reshape(-1, 1).repeat(h.shape[0], 1))
    return y


""" ====================== Q(c|X) ========================== """

Wqxh = xavier_init(size=[X_dim, h_dim])
Wqxh.name = 'Wqxh'
bqxh = Parameter(mindspore.ops.zeros(h_dim), requires_grad=True)
bqxh.name = 'bqxh'

Whc = xavier_init(size=[h_dim, 10])
Whc.name = 'Whc'
bhc = Parameter(mindspore.ops.zeros(10), requires_grad=True)
bhc.name = 'bhc'


def Q(X):
    h = mindspore.ops.relu(X @ Wqxh + bqxh.reshape(-1, 1).repeat(X.shape[0], 1))
    c = mindspore.ops.softmax(h @ Whc + bhc.reshape(1, -1).repeat(h.shape[0], 0))
    return c


G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]
Q_params = [Wqxh, bqxh, Whc, bhc]
params = G_params + D_params + Q_params


""" ===================== TRAINING ======================== """

D_loss_fn = lambda D_real, D_fake: -mindspore.ops.mean(mindspore.ops.log(D_real + 1e-8) + mindspore.ops.log(1 - D_fake + 1e-8))
G_loss_fn = lambda D_fake: -mindspore.ops.mean(mindspore.ops.log(D_fake + 1e-8))
crossent_loss_fn = lambda Q_c_given_x: mindspore.ops.mean(-mindspore.ops.sum(c * mindspore.ops.log(Q_c_given_x + 1e-8), dim=1))

D_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D_params)
G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G_params)
crossent_grad_fn = mindspore.value_and_grad(crossent_loss_fn, None, G_params + Q_params)

G_solver = mindspore.nn.Adam(G_params, learning_rate=1e-3)
D_solver = mindspore.nn.Adam(D_params, learning_rate=1e-3)
Q_solver = mindspore.nn.Adam(G_params + Q_params, learning_rate=1e-3)


def sample_c(size):
    c = np.random.multinomial(1, 10*[0.1], size=size)
    c = Parameter(mindspore.Tensor.from_numpy(c.astype('float32')))
    return c

data_loader = train_dataset.create_tuple_iterator()

for it in range(100000):
    # Sample data
    try:
        X, _ = next(data_loader)
        if X.shape[0] != mb_size:
            continue
    except StopIteration:
        data_loader = train_dataset.create_tuple_iterator()
        continue
    X = X.flatten().view(X.shape[0], -1)
    X = Parameter(X, requires_grad=False)
    z = Parameter(mindspore.ops.randn(mb_size, Z_dim), requires_grad=False)
    c = sample_c(mb_size)

    # Dicriminator forward-loss-backward-update
    G_sample = G(z, c)
    D_real = D(X)
    D_fake = D(G_sample)

    D_loss, D_grad = D_grad_fn(D_real, D_fake)
    D_solver(D_grad)

    # Generator forward-loss-backward-update
    G_sample = G(z, c)
    D_fake = D(G_sample)

    G_loss, G_grad = G_grad_fn(D_fake)
    G_solver(G_grad)

    # Q forward-loss-backward-update
    G_sample = G(z, c)
    Q_c_given_x = Q(G_sample)

    mi_loss, mi_grad = crossent_grad_fn(Q_c_given_x)
    Q_solver(mi_grad)

    # Print and plot every now and then
    if it % 1000 == 0:
        idx = np.random.randint(0, 10)
        c = np.zeros([mb_size, 10])
        c[range(mb_size), idx] = 1
        c = Parameter(mindspore.Tensor.from_numpy(c.astype('float32')))
        samples = G(z, c).numpy()[:16]

        print('Iter-{}; D_loss: {}; G_loss: {}; Idx: {}'
              .format(it, D_loss.numpy(), G_loss.numpy(), idx))

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
