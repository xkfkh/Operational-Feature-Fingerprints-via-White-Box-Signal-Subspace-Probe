# Data Preparation

The final paper datasets are:

- Amazon-Computers
- Amazon-Photo
- Chameleon
- Cornell
- Texas
- Wisconsin

The original paper describes these as PyTorch Geometric datasets. The local experiment code expects data under a local data root, typically planetoid/data.

Large data cache files such as .pt, .npz, .pkl, .npy, and downloaded dataset caches are intentionally not included in this Git repository.

Use the --data-root argument in scripts when your local data directory is different.

Squirrel and actor are optional/intermediate datasets and are not part of the final six-dataset paper tables.
