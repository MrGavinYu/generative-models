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
y_dim = 10
h_dim = 128
cnt = 0
lr = 1e-3
eps = 1e-8


G_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(z_dim + y_dim, h_dim),
    mindspore.nn.PReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)
G_.update_parameters_name('G_')

def G(z, c):
    inputs = mindspore.ops.cat([z, c], axis=1)
    return G_(inputs)


D_shared = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.PReLU()
)
D_shared.update_parameters_name('D_shared')

D_gan = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(h_dim, 1),
    mindspore.nn.Sigmoid()
)
D_gan.update_parameters_name('D_gan')

D_aux = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(h_dim, y_dim),
)
D_aux.update_parameters_name('D_aux')


def D(X):
    h = D_shared(X)
    return D_gan(h), D_aux(h)


nets = [G_, D_shared, D_gan, D_aux]

G_params = G_.trainable_params()
D_params = (list(D_shared.trainable_params()) + list(D_gan.trainable_params()) +
            list(D_aux.trainable_params()))


G_solver = mindspore.nn.Adam(G_params, learning_rate=lr)
D_solver = mindspore.nn.Adam(D_params, learning_rate=lr)
DC_loss_fn = lambda D_loss, C_loss: -(D_loss + C_loss)
DC_grad_fn = mindspore.value_and_grad(DC_loss_fn, None, D_params)
GC_loss_fn = lambda G_loss, C_loss: -(G_loss + C_loss)
GC_grad_fn = mindspore.value_and_grad(GC_loss_fn, None, G_params)


data_loader = train_dataset.create_tuple_iterator()

for it in tqdm(range(100000)):
    # Sample data
    try:
        X, y = next(data_loader)
    except StopIteration:
        data_loader = train_dataset.create_tuple_iterator()
        continue
    c = mindspore.ops.one_hot(Tensor(y, mindspore.int32), depth, on_value, off_value, axis=-1)
    X = Parameter(X, requires_grad=False)
    X = X.flatten().view(X.shape[0], -1)
    # c is one-hot
    c = Parameter(c.astype('float32'), requires_grad=False)
    # y_true is not one-hot (requirement from nn.cross_entropy)
    y_true = Parameter(y.view(-1, 1).argmax(axis=1).astype('int'), requires_grad=False)
    # z noise
    z = Parameter(mindspore.ops.randn(X.shape[0], z_dim), requires_grad=False)

    """ Discriminator """
    G_sample = G(z, c)
    D_real, C_real = D(X)
    D_fake, C_fake = D(G_sample)

    # GAN's D loss
    D_loss = mindspore.ops.mean(mindspore.ops.log(D_real + eps) + mindspore.ops.log(1 - D_fake + eps))
    # Cross entropy aux loss
    C_loss = -mindspore.ops.cross_entropy(C_real, y_true) - mindspore.ops.cross_entropy(C_fake, y_true)

    # Maximize

    DC_loss, DC_grad = DC_grad_fn(D_loss, C_loss)
    D_solver(DC_grad)

    """ Generator """
    G_sample = G(z, c)
    D_fake, C_fake = D(G_sample)
    _, C_real = D(X)

    # GAN's G loss
    G_loss = mindspore.ops.mean(mindspore.ops.log(D_fake + eps))
    # Cross entropy aux loss
    C_loss = -mindspore.ops.cross_entropy(C_real, y_true) - mindspore.ops.cross_entropy(C_fake, y_true)

    # Maximize
    GC_loss, GC_grad = GC_grad_fn(G_loss, C_loss)
    G_solver(GC_grad)

    # Print and plot every now and then
    if it % 1000 == 0:
        idx = np.random.randint(0, 10)
        c = np.zeros([16, y_dim])
        c[range(16), idx] = 1
        c = Parameter(c.astype('float32'), requires_grad=False)

        z = mindspore.ops.randn(16, z_dim)

        samples = G(z, c).numpy()

        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; Idx: {}'
              .format(it, -D_loss.numpy(), -G_loss.numpy(), idx))

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
