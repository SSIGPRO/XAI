import torch

B = 2**4
n_classes = 100
n_clusters = 50

seed = 42
torch.manual_seed(seed)

## Single sample single class

m = torch.rand(n_clusters)
mu = torch.rand(n_clusters)
sigmainv = torch.rand(n_clusters, n_clusters)

delta = m - mu

d_m = delta @ sigmainv @ delta

d_m_einsum = torch.einsum('i, ij, j ->', [delta, sigmainv, delta])

print(d_m_einsum == d_m)

## Single sample multiple classes

m = torch.rand(n_clusters)
mu = torch.rand(n_classes, n_clusters)
sigmainv = torch.rand(n_classes, n_clusters, n_clusters)

delta = m.unsqueeze(0) - mu

d_m = []

for c in range(n_classes):
    d_m.append(torch.einsum('i, ij, j ->', [delta[c], sigmainv[c], delta[c]]))
d_ = torch.stack(d_m)

d_m_einsum = torch.einsum('ki, kij, kj -> k', [delta, sigmainv, delta])

print(d_ == d_m_einsum)

## batch multiple classes
m = torch.rand(B, n_clusters)
mu = torch.rand(n_classes, n_clusters)
sigmainv = torch.rand(n_classes, n_clusters, n_clusters)

## Mathematical error due to rounding if u use higher precision the are identical

# m = m.to(torch.float64)
# mu = mu.to(torch.float64)
# sigmainv = sigmainv.to(torch.float64)

d_m = []

for l in range(B):
    
    delta = m[l].unsqueeze(0) - mu
    d_m.append(torch.einsum('ki, kij, kj -> k', [delta, sigmainv, delta]))

d_ = torch.stack(d_m)

delta = m.unsqueeze(1) - mu
d_m_einsum = torch.einsum('bki, kij, bkj -> bk', [delta, sigmainv, delta])

diff = d_m_einsum - d_
print("max abs:", diff.abs().max())
print("mean abs:", diff.abs().mean())
print("rmse:", diff.pow(2).mean().sqrt())
print(torch.allclose(d_m_einsum, d_, rtol=1e-4, atol=1e-6))