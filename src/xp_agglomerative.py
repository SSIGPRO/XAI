
import cuml
cuml.accel.install()

from cuml.cluster.agglomerative import AgglomerativeClustering

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1) Toy dataset (CPU)
X_cpu, y_true = make_blobs(n_samples=10, centers=3, cluster_std=0.7, random_state=42)
X_cpu = StandardScaler().fit_transform(X_cpu)

# 2) Fit cuML Agglomerative (no distances returned, but we get the merge structure)
agg = AgglomerativeClustering(
    n_clusters=None,          # build full tree
    linkage="single",
    distance_threshold=0.0    # forces full hierarchy construction
)
agg.fit(X_cpu)

# 3) Inspect children_ (shape: (n_samples-1, 2))
children = agg.children_
# If your environment returns a device array (rare with NumPy input), you might need:
# children = np.asarray(children)  # should work in most RAPIDS versions
print("Children array shape:", children.shape)
print(children[:5])

# 4) Reconstruct hierarchy levels (cluster members at each merge)
n_samples = X_cpu.shape[0]
clusters = {i: [i] for i in range(n_samples)}   # cluster_id -> member sample indices
levels = []

for new_cluster_id, (a, b) in enumerate(children.astype(int), start=n_samples):
    members = clusters[int(a)] + clusters[int(b)]
    clusters[new_cluster_id] = members
    levels.append((new_cluster_id, members))

# 5) Print first few hierarchy levels
for cid, members in levels[:5]:
    print(f"Cluster {cid}: merged -> {members}")

