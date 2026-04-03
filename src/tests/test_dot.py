import torch

import torch

B = 2**4
n_classes = 100
n_clusters = 50

seed = 42
torch.manual_seed(seed)

## Single sample single class

m = torch.rand(n_clusters)
mu = torch.rand(n_clusters)

d = m @ mu
d_einsum = torch.einsum('i, i ->', m, mu)
print(d, d_einsum, d==d_einsum)

## Single sample multiple class

m = torch.rand(n_clusters)
mu = torch.rand(n_classes, n_clusters)

d = mu @ m
d_einsum = torch.einsum('ij, j ->i', mu, m)
print(d.shape, d_einsum.shape, d==d_einsum)

## Batch multiple class

m = torch.rand(B, n_clusters)
mu = torch.rand(n_classes, n_clusters)

d = m @ mu.t()
d_einsum = torch.einsum('ij, bj ->bi', mu, m)
print(d.shape, d_einsum.shape)
print(d==d_einsum)