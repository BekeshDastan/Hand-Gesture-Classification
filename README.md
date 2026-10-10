# Hand Gesture Classification

A computer vision project that classifies three hand gestures: **thumbs up**, **thumbs down**, and **stop**. It compares a baseline CNN, a regularized CNN, and transfer learning with MobileNetV2 using TensorFlow and Keras.

The repository includes 450 images, annotated hand crops stored in notebook metadata, two executed notebooks, three trained models, and their training histories.

## Dataset

| Class | Personal originals | Augmented variants | External images | Total |
| --- | ---: | ---: | ---: | ---: |
| `thumbs_up` | 9 | 66 | 75 | 150 |
| `thumbs_down` | 7 | 68 | 75 | 150 |
| `stop` | 12 | 63 | 75 | 150 |
| **Total** | **28** | **197** | **225** | **450** |

External photographs come from HaGRID, as documented in the preprocessing notebook. Their hand boxes come from the original annotations. Personal photographs use manually annotated boxes, with corresponding boxes for their saved augmented variants. Both notebooks embed crop coordinates and SHA-256 checksums for all images.

The external images use a fixed stratified split with seed **42**: 135 for training, 45 for validation, and 45 for testing. All 225 personal originals and variants stay in training. The resulting split is **360 training / 45 validation / 45 test** images, with balanced gesture classes in each split. Training weights give the personal and external sources equal total influence.

## Preprocessing and models

Each annotated hand is cropped to a square with 18% context on each side of its longest dimension, resized to **128 x 128 RGB**, and normalized to `[0, 1]`. MobileNetV2 rescales inputs to `[-1, 1]` inside the model. Training applies mild augmentation; vertical flips are avoided because they can change thumb direction.

| Model | Approach | Saved checkpoint |
| --- | --- | --- |
| Baseline CNN | Three convolution blocks and a compact dense classifier | `models/baseline_cnn.keras` |
| Improved CNN | Four convolution blocks, batch normalization, L2 regularization, and dropout | `models/improved_cnn.keras` |
| MobileNetV2 | Frozen ImageNet backbone with a new three-class classifier | `models/mobilenet_v2.keras` |

Early stopping restores the checkpoint with the best validation accuracy. A learning rate scheduler monitors validation loss. Model selection uses validation accuracy.

## Recorded results

The executed project notebook reports these results on the same 45 external test images:

| Model | Validation accuracy | Test accuracy | Correct test predictions | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline CNN | 82.22% | 86.67% | 39 / 45 | 0.864 |
| Improved CNN | **93.33%** | **88.89%** | **40 / 45** | **0.890** |
| MobileNetV2 | 91.11% | 86.67% | 39 / 45 | 0.867 |

The improved CNN is selected by validation accuracy. Classification reports, confusion matrices, and learning curves are included in the notebook. Training histories and run settings are saved in `models/training_history.json`.

## Project structure

```text
Hand-Gesture-Classification/
|-- dataset/                       # 28 personal photographs, organized by class
|-- augmented/                     # 197 saved personal variants, organized by class
|-- external/                      # 225 HaGRID photographs, organized by class
|-- notebooks/
|   |-- 01_preprocessing.ipynb      # Counts, annotations, hand crops, normalization
|   `-- 02_gesture_project.ipynb    # Training, model comparison, evaluation, inference
|-- models/
|   |-- baseline_cnn.keras
|   |-- improved_cnn.keras
|   |-- mobilenet_v2.keras
|   `-- training_history.json
|-- .gitignore
`-- README.md
```

## Run locally

The saved notebook run used **Python 3.13**, **TensorFlow 2.21.0**, and **Keras 3.15.1**. The commands below use PowerShell on Windows:

```powershell
git clone https://github.com/BekeshDastan/Hand-Gesture-Classification.git
cd Hand-Gesture-Classification
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install tensorflow==2.21.0 keras==3.15.1 numpy pandas pillow matplotlib scikit-learn jupyterlab
.\.venv\Scripts\python.exe -m jupyter lab
```

1. Open `notebooks/01_preprocessing.ipynb` and run its cells in order to inspect the data and preprocessing.
2. Open `notebooks/02_gesture_project.ipynb`. **Before running its cells, set `RUN_TRAINING = False` in the settings cell** to load the supplied models. The committed setting is `True`, even though the notebook introduction describes loading as the default.
3. Run the project notebook from top to bottom to load the checkpoints, display training curves, compare models, and evaluate the test split. Both notebooks support starting from the project root or the `notebooks` folder.

To retrain, set `RUN_TRAINING = True`. Training updates the three model checkpoints and the history file. MobileNetV2 may download ImageNet weights on its first training run. Results can vary across runtimes and hardware.

## Classify a new image

After running the project notebook, call its inference helper on a photograph already cropped around one hand:

```python
gesture, confidence = predict_cropped_hand("path/to/cropped_hand.jpg")
print(gesture, confidence)
```

The helper uses the model selected on validation data. Output labels follow this order: `thumbs_up`, `thumbs_down`, `stop`.

## Evaluation limits

The models classify a hand whose location is already known. Full-frame photographs need a hand detector or a manually supplied crop before classification. The 197 personal variants derive from only 28 original photographs and do not represent independent participants. Validation and testing use external images only, and the test set has just 45 examples. This test split was inspected earlier in the project, so a fresh collection is needed for an independent follow-up evaluation.
