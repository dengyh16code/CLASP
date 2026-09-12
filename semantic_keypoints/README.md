# Semantic keypoint extraction

This module extracts semantic keypoints from a clothes image using OWL-ViT,
SAM, and SD-DINO. Complete the model-service and API configuration steps in the
main [README](../README.md) before running the example.

## Run the example

From the repository root, activate the `clasp` environment and load the project
paths:

```bash
conda activate clasp
source prepare_1.0.sh
```

Run the provided `semantic_keypoints/test.png` example from the repository root:

```bash
python semantic_keypoints/semantic_keypoints.py
```

The annotated result is saved to
`semantic_keypoints/extracted_keypoints.png`.
