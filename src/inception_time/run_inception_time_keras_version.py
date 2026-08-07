import sys
from pathlib import Path

import numpy as np
from tensordict import PersistentTensorDict

# Add peepholelib to Python path
sys.path.insert(
    0,
    (Path.home() / "repos/peepholelib").as_posix()
)

from classifier_inception import Classifier_INCEPTION


# ============================================================
# Paths
# ============================================================

# Location of parsed UEA datasets
PARSED_DATA_ROOT = Path.cwd() / "../data/datasets"

# Select ONE UEA sub-dataset
DATASET_NAME = "ArticularyWordRecognition"

# Location where InceptionTime results will be saved
OUTPUT_ROOT = Path.cwd() / "../results/inception_time"


# ============================================================
# InceptionTime parameters
# ============================================================
BATCH_SIZE = 8
LEARNING_RATE = 0.0005
NB_FILTERS = 16
DEPTH = 4
KERNEL_SIZE = 41
BOTTLENECK_SIZE = 16
NB_EPOCHS = 200
USE_RESIDUAL = True
USE_BOTTLENECK = True


# ============================================================
# Helper function to load one parsed split
# ============================================================

def load_split(split_directory):

    split_directory = Path(split_directory)

    # Each split contains only chunk_0
    chunk_path = split_directory / "chunk_0"

    # --------------------------------------------------------
    # Check chunk
    # --------------------------------------------------------

    if not chunk_path.exists():

        raise FileNotFoundError(
            f"Could not find:\n{chunk_path}"
        )

    print("\n" + "-" * 70)
    print(f"Loading: {chunk_path}")
    print("-" * 70)

    # --------------------------------------------------------
    # Open PersistentTensorDict
    # --------------------------------------------------------

    ptd = PersistentTensorDict.from_h5(
        str(chunk_path),
        mode="r"
    )

    print("Keys:", list(ptd.keys()))

    # --------------------------------------------------------
    # Check required keys
    # --------------------------------------------------------

    if "timeseries" not in ptd.keys():

        raise KeyError(
            f"'timeseries' key not found in {chunk_path}"
        )

    if "label" not in ptd.keys():

        raise KeyError(
            f"'label' key not found in {chunk_path}"
        )

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    X = ptd["timeseries"].numpy()
    y = ptd["label"].numpy()

    print("timeseries:")
    print("  shape :", X.shape)
    print("  dtype :", X.dtype)

    print("label:")
    print("  shape :", y.shape)
    print("  dtype :", y.dtype)

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    if X.ndim != 3:

        raise ValueError(
            f"Expected timeseries shape "
            f"(N, channels, time_steps), "
            f"but received {X.shape}"
        )

    if y.ndim != 1:

        raise ValueError(
            f"Expected label shape (N,), "
            f"but received {y.shape}"
        )

    if X.shape[0] != y.shape[0]:

        raise ValueError(
            "Number of timeseries samples and labels "
            "does not match."
        )

    return X, y


# ============================================================
# Locate parsed dataset
# ============================================================

dataset_root = (
    PARSED_DATA_ROOT / DATASET_NAME
)

train_directory = (
    dataset_root / "dss.UEA-train"
)

validate_directory = (
    dataset_root / "dss.UEA-val"
)

test_directory = (
    dataset_root / "dss.UEA-test"
)


# ============================================================
# Display configuration
# ============================================================

print("\n" + "=" * 70)
print("InceptionTime - UEA Dataset")
print("=" * 70)

print("Dataset:")
print(DATASET_NAME)

print("\nParsed dataset root:")
print(dataset_root)

print("\nTrain:")
print(train_directory / "chunk_0")

print("\nValidation:")
print(validate_directory / "chunk_0")

print("\nTest:")
print(test_directory / "chunk_0")


# ============================================================
# Check directories
# ============================================================

for directory in [
    train_directory,
    validate_directory,
    test_directory
]:

    if not directory.exists():

        raise FileNotFoundError(
            f"\nDirectory does not exist:\n{directory}"
        )


# ============================================================
# Load train / validation / test
# ============================================================

X_train, y_train = load_split(
    train_directory
)

X_validate, y_validate = load_split(
    validate_directory
)

X_test, y_test = load_split(
    test_directory
)


# ============================================================
# Display original data shapes
# ============================================================

print("\n" + "=" * 70)
print("Original ParsedDataset shapes")
print("=" * 70)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_validate:", X_validate.shape)
print("y_validate:", y_validate.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# Verify dimensional consistency
# ============================================================

train_channels = X_train.shape[1]
train_time_steps = X_train.shape[2]

if X_validate.shape[1] != train_channels:

    raise ValueError(
        "Train and validation data have different "
        "numbers of channels."
    )

if X_test.shape[1] != train_channels:

    raise ValueError(
        "Train and test data have different "
        "numbers of channels."
    )

if X_validate.shape[2] != train_time_steps:

    raise ValueError(
        "Train and validation data have different "
        "numbers of time steps."
    )

if X_test.shape[2] != train_time_steps:

    raise ValueError(
        "Train and test data have different "
        "numbers of time steps."
    )


# ============================================================
# Convert time-series layout
# ============================================================

# ParsedDataset format:
#
#     (samples, channels, time_steps)
#
# Example:
#
#     (25, 11, 217)
#
# InceptionTime / Keras Conv1D format:
#
#     (samples, time_steps, channels)
#
# Therefore:
#
#     (25, 11, 217)
#
# becomes:
#
#     (25, 217, 11)

print("\n" + "=" * 70)
print("Converting time-series layout")
print("=" * 70)

X_train = np.transpose(
    X_train,
    (0, 2, 1)
)

X_validate = np.transpose(
    X_validate,
    (0, 2, 1)
)

X_test = np.transpose(
    X_test,
    (0, 2, 1)
)

print("X_train:", X_train.shape)
print("X_validate:", X_validate.shape)
print("X_test:", X_test.shape)


# ============================================================
# Convert labels to NumPy integer arrays
# ============================================================

y_train = np.asarray(
    y_train,
    dtype=np.int64
)

y_validate = np.asarray(
    y_validate,
    dtype=np.int64
)

y_test = np.asarray(
    y_test,
    dtype=np.int64
)


# ============================================================
# Find all classes
# ============================================================

all_labels = np.concatenate(
    [
        y_train,
        y_validate,
        y_test
    ]
)

classes = np.unique(
    all_labels
)

nb_classes = len(classes)

print("\n" + "=" * 70)
print("Class information")
print("=" * 70)

print("Classes:", classes)
print("Number of classes:", nb_classes)


# ============================================================
# Map labels to consecutive integers
# ============================================================

class_to_index = {
    label: index
    for index, label in enumerate(classes)
}

y_train = np.array(
    [
        class_to_index[label]
        for label in y_train
    ],
    dtype=np.int64
)

y_validate = np.array(
    [
        class_to_index[label]
        for label in y_validate
    ],
    dtype=np.int64
)

y_test = np.array(
    [
        class_to_index[label]
        for label in y_test
    ],
    dtype=np.int64
)


# ============================================================
# Convert labels to one-hot encoding
# ============================================================

y_train = np.eye(
    nb_classes,
    dtype=np.float32
)[y_train]

y_validate = np.eye(
    nb_classes,
    dtype=np.float32
)[y_validate]

y_test = np.eye(
    nb_classes,
    dtype=np.float32
)[y_test]

print("\nOne-hot label shapes:")

print("y_train:", y_train.shape)
print("y_validate:", y_validate.shape)
print("y_test:", y_test.shape)


# ============================================================
# Determine InceptionTime input shape
# ============================================================

input_shape = X_train.shape[1:]

print("\n" + "=" * 70)
print("InceptionTime input")
print("=" * 70)

print("Input shape:", input_shape)
print("Number of classes:", nb_classes)


# ============================================================
# Create output directory
# ============================================================

output_directory = (
    OUTPUT_ROOT / DATASET_NAME
)

output_directory.mkdir(
    parents=True,
    exist_ok=True
)

# Classifier expects a string ending in "/"
output_directory = (
    str(output_directory) + "/"
)


# ============================================================
# Create InceptionTime classifier
# ============================================================

print("\n" + "=" * 70)
print("Building InceptionTime")
print("=" * 70)

classifier = Classifier_INCEPTION(

    output_directory=output_directory,

    input_shape=input_shape,

    nb_classes=nb_classes,

    verbose=True,

    build=True,

    batch_size=BATCH_SIZE,

    lr=LEARNING_RATE,

    nb_filters=NB_FILTERS,

    use_residual=USE_RESIDUAL,

    use_bottleneck=USE_BOTTLENECK,

    depth=DEPTH,

    kernel_size=KERNEL_SIZE,

    nb_epochs=NB_EPOCHS,

    bottleneck_size=BOTTLENECK_SIZE

)


# ============================================================
# Train InceptionTime
# ============================================================

print("\n" + "=" * 70)
print("Starting InceptionTime training")
print("=" * 70)

results = classifier.fit(

    # Training data
    X_train,
    y_train,

    # Validation data
    X_validate,
    y_validate,

    # Test data
    X_test,
    y_test

)


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 70)
print("Final Results")
print("=" * 70)

print(results)

print("\nDataset:", DATASET_NAME)

print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Validation samples:",
    X_validate.shape[0]
)

print(
    "Test samples:",
    X_test.shape[0]
)

print("\nResults saved at:")

print(output_directory)
