import mindspore
import mindspore.nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
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
X_dim = 28*28
Z_dim = 100
y_dim = 10
h_dim = 128
cnt = 0
lr = 1e-3


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Parameter(mindspore.ops.randn(*size) * xavier_stddev, requires_grad=True)


""" ==================== GENERATOR ======================== """

Wzh = xavier_init(size=[Z_dim + y_dim, h_dim])
Wzh.name = 'Wzh'
bzh = Parameter(mindspore.ops.zeros(h_dim), requires_grad=True)
bzh.name = 'bzh'

Whx = xavier_init(size=[h_dim, X_dim])
Whx.name = 'Whx'
bhx = Parameter(mindspore.ops.zeros(X_dim), requires_grad=True)
bhx.name = 'bhx'

def G(z, c):
    inputs = mindspore.ops.cat([z, c], axis=1)
    h = mindspore.ops.relu(inputs @ Wzh + bzh.tile((inputs.shape[0], 1)))
    X = mindspore.ops.sigmoid(h @ Whx + bhx.tile((h.shape[0], 1)))
    return X


""" ==================== DISCRIMINATOR ======================== """

Wxh = xavier_init(size=[X_dim + y_dim, h_dim])
Wxh.name = 'Wxh'
bxh = Parameter(mindspore.ops.zeros(h_dim), requires_grad=True)
bxh.name = 'bxh'

Why = xavier_init(size=[h_dim, 1])
Why.name = 'Why'
bhy = Parameter(mindspore.ops.zeros(1), requires_grad=True)
bhy.name = 'bhy'

def D(X, c):
    inputs = mindspore.ops.cat([X, c], axis=1)
    h = mindspore.ops.relu(inputs @ Wxh + bxh.tile((inputs.shape[0], 1)))
    y = mindspore.ops.sigmoid(h @ Why + bhy.tile((h.shape[0], 1)))
    return y


G_params = [Wzh, bzh, Whx, bhx]
D_params = [Wxh, bxh, Why, bhy]
params = G_params + D_params


""" ===================== TRAINING ======================== """
G_solver = mindspore.nn.Adam(G_params, learning_rate=1e-3)
D_solver = mindspore.nn.Adam(D_params, learning_rate=1e-3)

data_loader = train_dataset.create_tuple_iterator()


def D_loss_fn(D_loss_real, D_loss_fake):
    return D_loss_real + D_loss_fake

D_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D_params)

for it in tqdm(range(100000)):
    # Sample data
    try:
        X, c = next(data_loader)
        c = mindspore.ops.one_hot(Tensor(c, mindspore.int32), depth, on_value, off_value, axis=-1)
    except StopIteration:
        data_loader = train_dataset.create_tuple_iterator()
        continue
    X = X.flatten().view(X.shape[0], -1)
    z = Parameter(mindspore.ops.randn(X.shape[0], Z_dim), requires_grad=False)
    X = Parameter(X, requires_grad=False)
    c = Parameter(c.astype('float32'), requires_grad=False)

    # Dicriminator forward-loss-backward-update
    G_sample = G(z, c)
    D_real = D(X, c)
    D_fake = D(G_sample, c)

    ones_label = Parameter(mindspore.ops.ones((X.shape[0], 1)))
    zeros_label = Parameter(mindspore.ops.zeros((X.shape[0], 1)))
    D_loss_real = mindspore.ops.binary_cross_entropy(D_real, ones_label)
    D_loss_fake = mindspore.ops.binary_cross_entropy(D_fake, zeros_label)
    D_loss = D_loss_real + D_loss_fake

    D_loss, D_grad = D_grad_fn(D_loss_real, D_loss_fake)
    D_solver(D_grad)

    # Generator forward-loss-backward-update
    z = Parameter(mindspore.ops.randn(X.shape[0], Z_dim), requires_grad=False)
    G_sample = G(z, c)
    D_fake = D(G_sample, c)

    G_loss_fn = lambda D_fake: mindspore.ops.binary_cross_entropy(D_fake, ones_label)
    G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G_params)
    G_loss, G_grad = G_grad_fn(D_fake)
    G_solver(G_grad)

    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {}; G_loss: {}'.format(it, D_loss.numpy(), G_loss.numpy()))

        c = np.zeros(shape=[mb_size, y_dim], dtype='float32')
        c[:, np.random.randint(0, 10)] = 1.
        c = Parameter(c, requires_grad=False)
        samples = G(z, c).numpy()[:16]

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
