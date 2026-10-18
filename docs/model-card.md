# Model card

## Task and architecture

Binary chest X-ray image classification: `Normal` (index 0) or `Pneumonia` (index 1). The architecture is ResNet18 with a one-channel first convolution and a two-output final linear layer. Inference runs on CPU with the same resize/grayscale/normalization used in the training script.

## Checkpoint

`models/aeriscan_best.pth` was recovered from the author's original local project. It contains a PyTorch state dictionary and is loaded using `weights_only=True`. The SHA-256 checksum is recorded in `models/SHA256SUMS`.

The checkpoint's architecture is verified by loading it and running an inference pass. This does not establish which exact training run produced it. A training manifest, seed, dataset checksum, training logs, and independently reproducible evaluation report are not available in the repository.

## Training code

The checked-in script starts from ImageNet ResNet18 weights, replaces the input and output layers, and trains for six epochs with Adam (`1e-4` learning rate, batch size 8). It uses weighted cross entropy, horizontal-flip augmentation, and lowest validation loss to select the best checkpoint. These are properties of the script, not a verified record of the recovered checkpoint's training run.

The expected data layout is ImageFolder with `NORMAL` and `PNEUMONIA` folders. The original repository contained duplicated JPEG dataset directories and macOS metadata. They are excluded from the current tree. The source URL, dataset version, patient split, and redistribution terms were not recorded, so they must be verified before redistributing data or reporting evaluation results. No dataset ownership or license is asserted here.

## Interpretation and limits

The predicted class is the largest softmax output. The original API risk labels are retained for compatibility: Normal maps to Low, and Pneumonia maps to High at a pneumonia score of at least 0.8, otherwise Moderate. These thresholds are heuristics, not validated clinical risk categories. Its score is not calibrated and is not a clinical risk estimate. Grad-CAM uses `layer4` activations and gradients for that predicted class, applies ReLU, normalizes the map, and returns a 224 × 224 transparent PNG. The frontend stretches the map to the displayed image, reversing the model's fixed-size resize.

The system does not validate image modality, patient age, image quality, or acquisition conditions. It cannot distinguish a valid chest X-ray from an arbitrary JPEG/PNG. Out-of-distribution inputs may receive high scores. Sample filenames are inherited from the project and are not independently verified labels.

Do not use the application for diagnosis, triage, treatment, or reassurance. Software tests cover checkpoint compatibility and API behavior; they do not measure clinical performance.

## Evaluation work still needed

Before claiming model quality, record the dataset source and permitted use, establish patient-separated train/validation/test splits, evaluate the untouched test split, and report confusion counts, sensitivity, specificity, and score calibration. Include the evaluation code, checkpoint checksum, dependency versions, and run configuration with the results.
