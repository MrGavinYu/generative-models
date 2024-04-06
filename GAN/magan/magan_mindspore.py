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
d_step = 3
lr = 1e-3
m = 5
n_iter = 1000
n_epoch = 1000
N = n_iter * mb_size  # N data per epoch


G = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(z_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)

# D is an autoencoder
D_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
)

# Energy is the MSE of autoencoder
def D(X):
    X_recon = D_(X)
    return mindspore.ops.sum((X - X_recon)**2, 1)

D_loss_fn = lambda D_real, D_fake: mindspore.ops.mean(D_real) + mindspore.ops.relu(m - mindspore.ops.mean(D_fake))
loss_fn = lambda X: mindspore.ops.mean(D(X))
G_loss_fn = lambda D_fake: 0.5 * mindspore.ops.mean((D_fake - 1)**2)

D_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D_.trainable_params())
grad_fn = mindspore.value_and_grad(loss_fn, None, D_.trainable_params())
G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G.trainable_params())

G_solver = mindspore.nn.Adam(G.trainable_params(), learning_rate=lr)
D_solver = mindspore.nn.Adam(D_.trainable_params(), learning_rate=lr)

data_loader = train_dataset.create_tuple_iterator()

for it in range(2*n_iter):
    try:
        X, _ = next(data_loader)
        if X.shape[0] != mb_size:
            continue
    except StopIteration:
        data_loader = train_dataset.create_tuple_iterator()
        continue
    X = X.flatten().view(X.shape[0], -1)
    X = Parameter(X, requires_grad=False)

    loss, grad = grad_fn(X)
    D_solver(grad)

    if it % 1000 == 0:
        print('Iter-{}; Pretrained D loss: {:.4}'.format(it, loss.numpy()))

# Initial margin, expected energy of real data
images = [imgs for imgs, labels in list(train_dataset)]
images = mindspore.ops.concat(images)
images = images.flatten().view(images.shape[0], -1)
m = mindspore.ops.mean(D(Parameter(images, requires_grad=False))).numpy()
m = float(m)
s_z_before = mindspore.Tensor.from_numpy(np.array([np.inf], dtype='float32'))


# GAN training
for t in range(n_epoch):
    s_x, s_z = mindspore.ops.zeros(1), mindspore.ops.zeros(1)

    for it in range(n_iter):
        # Sample data
        z = Parameter(mindspore.ops.randn(mb_size, z_dim))
        try:
            X, _ = next(data_loader)
            if X.shape[0] != mb_size:
                continue
        except StopIteration:
            data_loader = train_dataset.create_tuple_iterator()
            continue

        X = X.flatten().view(X.shape[0], -1)
        X = Parameter(X, requires_grad=False)
        z = Parameter(mindspore.ops.randn(mb_size, z_dim), requires_grad=False)
        # Dicriminator
        G_sample = G(z)
        D_real = D(X)
        D_fake = D(G_sample)

        D_loss, D_grad = D_grad_fn(D_real, D_fake)
        D_solver(D_grad)

        # Update real samples statistics
        s_x += mindspore.ops.sum(D_real)

        # Generator
        z = Parameter(mindspore.ops.randn(mb_size, z_dim))
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss, G_grad = G_grad_fn(D_fake)
        G_solver(G_grad)

        s_z += mindspore.ops.sum(D_fake)

    # Update margin
    if (((s_x[0] / N) < m) and (s_x[0] < s_z[0]) and (s_z_before[0] < s_z[0])):
        m = s_x[0] / N

    s_z_before = s_z

    # Convergence measure
    Ex = s_x[0] / N
    Ez = s_z[0] / N
    L = Ex + np.abs(Ex - Ez)

    # Visualize
    print('Epoch-{}; m = {:.4}; L = {:.4}'
          .format(t, m, L.numpy()))

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