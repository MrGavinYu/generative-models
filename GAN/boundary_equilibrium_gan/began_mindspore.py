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
d_step = 3
lr = 1e-3
m = 5
lam = 1e-3
k = 0
gamma = 0.5


G = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(z_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
    mindspore.nn.Sigmoid()
)
G.update_parameters_name('G')

D_ = mindspore.nn.SequentialCell(
    mindspore.nn.Dense(X_dim, h_dim),
    mindspore.nn.ReLU(),
    mindspore.nn.Dense(h_dim, X_dim),
)
D_.update_parameters_name('D_')

# D is an autoencoder, approximating Gaussian
def D(X):
    X_recon = D_(X)
    # Use Laplace MLE as in the paper
    return mindspore.ops.mean(mindspore.ops.sum(mindspore.ops.abs(X - X_recon), 1))

D_solver = mindspore.nn.Adam(D_.trainable_params(), learning_rate=lr)
D_loss_fn = lambda X, z_D: D(X) - k * D(G(z_D))
D_grad_fn = mindspore.value_and_grad(D_loss_fn, None, D_.trainable_params())
G_solver = mindspore.nn.Adam(G.trainable_params(), learning_rate=lr)
G_loss_fn = lambda z_G: D(G(z_G))
G_grad_fn = mindspore.value_and_grad(G_loss_fn, None, G.trainable_params())

data_loader = train_dataset.create_tuple_iterator()

for it in tqdm(range(1000000)):
    # Sample data
    try:
        X, _ = next(data_loader)
    except StopIteration:
        data_loader = train_dataset.create_tuple_iterator()
        continue
    X = X.flatten().view(X.shape[0], -1)
    # Dicriminator
    z_D = Parameter(mindspore.ops.randn(X.shape[0], z_dim), requires_grad=False)

    D_loss, D_grad = D_grad_fn(X, z_D)
    D_solver(D_grad)

    # Generator
    z_G = Parameter(mindspore.ops.randn(X.shape[0], z_dim), requires_grad=False)

    G_loss, G_grad = G_grad_fn(z_G)
    G_solver(G_grad)

    # Update k, the equlibrium
    k = k + lam * (gamma*D(X) - D(G(z_G)))
    k = k.numpy().item()  # k is variable, so unvariable it so that no gradient prop.

    # Print and plot every now and then
    if it % 1000 == 0:
        measure = D(X) + mindspore.ops.abs(gamma*D(X) - D(G(z_G)))

        print('Iter-{}; Convergence measure: {:.4}'
              .format(it, measure.numpy().item()))

        samples = G(z_G).numpy()[:16]

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
