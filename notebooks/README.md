# Manafold research notebooks

These notebooks record Manafold's research trajectory, documenting model iterations, design rationales, and experimental findings. They preserve the full context of development, including initial hypotheses, control baselines, negative results, and changing interpretations that formal papers usually exclude. All notebooks are self-contained and run directly from the repository root.

The suite contains four chronological research volumes. Volume 01 (`01_NBAC_Baselines_and_Taxonomy_Ceiling.ipynb`) evaluates baseline models A0 through A2++, establishing the 1-NN card overlap accuracy ceiling and baseline diagnostic corrections. Volume 02 (`02_Set_Transformers_and_Auxiliary_Losses.ipynb`) covers models A3 through A7, testing Set Transformer architectures, hypergeometric copy pooling, partial-view training, and auxiliary loss ablations. Volume 03 (`03_Hyperbolic_Product_Manifolds_and_PMI_Anchors.ipynb`) details models A8 through A10, evaluating card-archetype Positive Mutual Information evidence paths, product manifold formulations, and geometric ablation results. Volume 04 (`04_Direct_Canonical_Ontology_and_Energy_A11.ipynb`) evaluates model A11, analyzing target graph induction, family projection policies, temperature calibration, confusion matrices, and Helmholtz Free Energy out-of-distribution detection.

Execute the notebooks sequentially, as each volume builds on earlier findings and data structures. Generated datasets and execution caches are written to `MANAFOLD_RESEARCH_WORKSPACE`, which defaults to the system temporary directory.

To run all notebooks from the repository root:

```bash
uv run --extra notebook jupyter execute notebooks/*.ipynb --inplace --timeout=7200
```

Database connection settings use the same environment variables as the dataset exporter in Manafold. The default `full` research profile runs 40 training epochs across every registered temporal window.

When contributing to these notebooks, you can run a quick verification check by setting `MANAFOLD_RESEARCH_PROFILE=smoke`. This profile generates complete temporal windows but limits model fitting to five optimizer steps. Set `MANAFOLD_RESEARCH_REUSE=0` to force all steps to re-execute from scratch.
