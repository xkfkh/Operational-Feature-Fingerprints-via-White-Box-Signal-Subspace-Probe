White-box Graph Mechanism Atlas key-data-only package

This slim package excludes PNG images and large node-level barcode files.

Included key files per dataset:

1. whitebox_dataset_summary_by_split.csv
   Main per-dataset mechanism summary.

2. A_graph_signal_block_fingerprint.csv
   Graph signal block fingerprint:
   I_X, I_PrX, I_Pr2X, I_Pr3X,
   I_XminusPrX, I_PrXminusPr2X,
   I_PsX, I_Ps2X, I_XminusPsX.

3. B_class_signal_glyph.csv
   Class-level Raw / Low-pass / High-pass / T / S signal data.

4. D_class_geometry_complexity.csv
   Class PCA intrinsic dimension C_c.

5. D_subspace_overlap_constellation.csv
   Pairwise class subspace overlap O_ab and confusion.

6. E_error_signal_shift.csv
   Delta_b = P_b_wrong - P_b_correct.

7. F_split_mechanism_stability_strip.csv
   Mechanism strip values. Since splits=1, this is single-split summary.

8. node_mechanism_type_accuracy.csv
   Accuracy grouped by inferred node mechanism type.

Scripts:
- scripts/run_whitebox_graph_mechanism_atlas.py
  Main feature/statistics script for reproduction.

Images are intentionally excluded.
Large node-level file A_graph_signal_barcode_nodes.csv is intentionally excluded.

