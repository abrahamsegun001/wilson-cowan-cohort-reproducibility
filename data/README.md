# Data layout

The raw BraiDyn-BC imaging data are not redistributed in this repository. Obtain the source dataset from its public archive and run the derivative-aware preprocessing pipeline under `the companion derivative-aware preprocessing repository`.

The cohort inference scripts expect the production 64 x 64 linear-preprocessed chunks in:

```text
data/processed_64_linear/
  sub-GC48_ses-day0_chunk-00_dfof.npz
  ...
```

`preprocessing_manifest_40.csv` fixes the 40 recording/task mapping used in the manuscript.

The full release should also contain the derived model-parameter tables collected by `scripts/collect_release_outputs.sh`. These are lightweight derived data and can be redistributed with the code even though the raw imaging files are not bundled.
