# Segmentation baselines

This directory contains the source-only comparison implementations used with
CGMFNet. Each baseline is self-contained and has its own README, dependencies,
training script, evaluation script, dataset loader, and model package.

For a fair comparison, use the same train-validation split, input resolution,
augmentation policy, loss definition, class mapping, IoU convention, and timing
protocol across all models.
