White-box Graph Mechanism Atlas submission package

This package contains:
1. results/
   Per-dataset A-F white-box mechanism figures and CSV files.

2. final_summary/
   7-dataset merged white-box mechanism summaries.

3. scripts/
   Main statistics script:
   - run_whitebox_graph_mechanism_atlas.py

   Parallel launcher:
   - run_whitebox_atlas_6datasets_parallel_final.sh

4. logs/
   Running logs for each dataset.

5. code_snapshot/
   Minimal source code snapshot required for reproduction.

Datasets:
- chameleon
- amazon-computers
- amazon-photo
- cornell
- squirrel
- texas
- wisconsin

Figures:
A. Graph Signal Barcode
B. Class Signal Glyph
C. Geometry-Boundary Mechanism Plane
D. Subspace Constellation
E. Error Signal Shift
F. Split Mechanism Stability Strip

Important note:
This white-box atlas analysis does not depend on E08.
It reruns the algorithm once per dataset, selects the best config by validation only,
and then records side-channel statistics after the best config is fixed.

Reproduce one dataset example:

cd code_snapshot

export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

python scripts/run_whitebox_graph_mechanism_atlas.py \
  --data-root /path/to/planetoid/data \
  --dataset cornell \
  --grid default \
  --splits 1 \
  --seed0 20260419 \
  --out-dir ./reproduce_cornell_atlas

For large datasets, use fast grid:

python scripts/run_whitebox_graph_mechanism_atlas.py \
  --data-root /path/to/planetoid/data \
  --dataset amazon-computers \
  --grid fast \
  --splits 1 \
  --seed0 20260419 \
  --out-dir ./reproduce_amazon_computers_atlas

Note:
Since splits=1, F is a single-split mechanism strip.
True split stability requires running with --splits > 1.

