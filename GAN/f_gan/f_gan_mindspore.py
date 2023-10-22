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
# y_train = mindspore.ops.one_hot(Tensor(y_train, mindspore.int32), depth, on_value, off_value, axis=-1)
X_dim = 28*28
z_dim = 10
h_dim = 128
cnt = 0
lr = 1e-3


def log(x):
    return mindspore.ops.log(x + 1e-8)


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
)


def reset_grad():
    G.zero_grad()
    D.zero_grad()

def D_loss_fn(z, X):
    G_sample = G(z)
    D_real = D(X)
    D_fake = D(G_sample)
    D_loss = -(mindspore.ops.mean(-mindspore.ops.exp(D_real)) - mindspore.ops.mean(-1 - D_fake))
    return D_loss

def G_loss_fn(z):
    G_sample = G(z)
    D_fake = D(G_sample)
    G_loss = -mindspore.ops.mean(-1 - D_fake)
    return G_loss

G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G.trainable_params())
D_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D.trainable_params())

G_solver = mindspore.nn.Adam(G.trainable_params(), learning_rate=lr)
D_solver = mindspore.nn.Adam(D.trainable_params(), learning_rate=lr)

data_loader = train_dataset.create_tuple_iterator()

for it in tqdm(range(1000000)):
    # Sample data
    z = Parameter(mindspore.ops.randn(mb_size, z_dim), requires_grad=False)
    try:
        X, _ = next(data_loader)
    except StopIteration:
        data_loader = train_dataset.create_tuple_iterator()
        continue
    X = X.flatten().view(X.shape[0], -1)
    X = Parameter(X, requires_grad=False)

    # Dicriminator
    D_loss, D_grad = D_grad_fn(z, X)
    D_solver(D_grad)
    # Uncomment D_loss and its respective G_loss of your choice
    # ---------------------------------------------------------

    """ Total Variation """
    # D_loss = -(mindspore.mean(0.5 * mindspore.tanh(D_real)) -
    #            mindspore.mean(0.5 * mindspore.tanh(D_fake)))
    """ Forward KL """
    # D_loss = -(mindspore.mean(D_real) - mindspore.mean(mindspore.exp(D_fake - 1)))
    """ Reverse KL """
    # D_loss = -(mindspore.ops.mean(-mindspore.ops.exp(D_real)) - mindspore.ops.mean(-1 - D_fake))
    """ Pearson Chi-squared """
    # D_loss = -(mindspore.mean(D_real) - mindspore.mean(0.25*D_fake**2 + D_fake))
    """ Squared Hellinger """
    # D_loss = -(mindspore.mean(1 - mindspore.exp(D_real)) -
    #            mindspore.mean((1 - mindspore.exp(D_fake)) / (mindspore.exp(D_fake))))

    # Generator
    G_loss, G_grad = G_grad_fn(z)
    G_solver(G_grad)
    """ Total Variation """
    # G_loss = -mindspore.mean(0.5 * mindspore.tanh(D_fake))
    """ Forward KL """
    # G_loss = -mindspore.mean(mindspore.exp(D_fake - 1))
    """ Reverse KL """
    # G_loss = -mindspore.ops.mean(-1 - D_fake)
    """ Pearson Chi-squared """
    # G_loss = -mindspore.mean(0.25*D_fake**2 + D_fake)
    """ Squared Hellinger """
    # G_loss = -mindspore.mean((1 - mindspore.exp(D_fake)) / (mindspore.exp(D_fake)))


    # Print and plot every now and then
    if it % 1000 == 0:
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}'
              .format(it, D_loss.numpy(), G_loss.numpy()))

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
