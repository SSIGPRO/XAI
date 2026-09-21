import sys
import tensorflow as tf
import torch
import numpy as np
from pathlib import Path
from functools import partial

sys.path.insert(0, (Path.home() / "repos/peepholelib").as_posix())

from peepholelib.datasets.UEAdataset import TSDataWrap
from peepholelib.datasets.parsedDataset import ParsedDataset
from classifier_inception import Classifier_INCEPTION
from cuda_selector import auto_cuda


def inception_classification_full(data, model):

    # Get the time series and labels
    x = data["timeseries"]
    y = data["label"]

    # PyTorch tensor -> NumPy
    x = x.detach().cpu().numpy()

    # PeepholeLib/UEA:
    # (batch, channels, time)
    #
    # Keras InceptionTime:
    # (batch, time, channels)
    x = np.transpose(x, (0, 2, 1))

    # NumPy -> TensorFlow
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # Inference
    predictions = model(x, training=False)

    # TensorFlow -> NumPy -> PyTorch
    predictions = torch.from_numpy(
        predictions.numpy()
    )

    # Predicted class
    predicted_class = torch.argmax(predictions, dim=1)

    # Return a DICTIONARY
    return {
        "output": predictions,
        "result": (predicted_class == y).long(),
    }

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('memory')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    # --------------------------------
    # Paths
    # --------------------------------

    uea_path = Path("/srv/newpenny/dataset/Multivariate_ts")
    ds_path = Path.cwd() / "../data/datasets"

    # Root directory containing the trained
    # InceptionTime models for all UEA datasets
    model_root = Path("/srv/newpenny/XAI/models/inception_time")

    seed = 29
    batch_size = 256
    verbose = True

    # --------------------------------
    # Process every UEA dataset
    # --------------------------------

    for dataset_dir in sorted(uea_path.iterdir()):

        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name

        print("\n" + "=" * 60)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 60)

        # --------------------------------
        # Dataset wrapper
        # --------------------------------

        dataset_wraps = {
            dataset_name: TSDataWrap(
                path=dataset_dir,
                seed=seed,
            )
        }

        # --------------------------------
        # Parsed dataset
        # --------------------------------

        dataset = ParsedDataset(
            path=ds_path / dataset_name
        )

        with dataset as ds:

            # --------------------------------
            # Parse UEA dataset
            # --------------------------------

            ds.parse_dataset(
                dataset_wraps=dataset_wraps,
                ds_samplers=None,
                keys_to_copy=["timeseries", "label"],
                batch_size=batch_size,
                n_threads=1,
                verbose=verbose,
            )

            # --------------------------------
            # Get dataset information
            # --------------------------------
            ts_dataset = dataset_wraps[dataset_name]

            # Number of classes
            n_classes = len(ts_dataset.label_map)

            # Input shape
            #
            # UEA data is stored as:
            #     (n_samples, n_channels, series_length)
            #
            # InceptionTime expects:
            #     (n_samples, series_length, n_channels)
            #
            # Therefore input_shape is:
            #     (series_length, n_channels)
            #
            # Use the underlying UEA-test dataset
            test_dataset = ts_dataset.__dataset__["UEA-test"]

            # UEA data shape:
            # (samples, channels, time_steps)
            #
            # InceptionTime expects:
            # (samples, time_steps, channels)

            input_shape = (
                test_dataset.X.shape[2],
                test_dataset.X.shape[1]
            )

            print(f"X shape     : {test_dataset.X.shape}")
            print(f"Input shape : {input_shape}")

            # --------------------------------
            # Dataset-specific model directory
            # --------------------------------

            dataset_model_dir = model_root / dataset_name

            weights_path = dataset_model_dir / "best_model.weights.h5"

            if not weights_path.exists():
                print(
                    f"WARNING: No trained weights found for "
                    f"{dataset_name}: {weights_path}"
                )
                continue

            # --------------------------------
            # Create InceptionTime model
            # --------------------------------

            inception_model = Classifier_INCEPTION(
                output_directory=str(dataset_model_dir) + "/",
                input_shape=input_shape,
                nb_classes=n_classes,
                verbose=verbose,
                build=True,
                batch_size=batch_size,
                lr=0.0005,
                nb_filters=16,
                use_residual=True,
                use_bottleneck=True,
                depth=4,
                kernel_size=41,
                nb_epochs=200,
                bottleneck_size=16,
            )

            # --------------------------------
            # Get the Keras model
            # --------------------------------

            model = inception_model.get_model()

            # --------------------------------
            # Load dataset-specific weights
            # --------------------------------

            model.load_weights(str(weights_path))

            print(f"Loaded weights: {weights_path}")

            # --------------------------------
            # Run inference
            # --------------------------------

            ds.parse_inference(
                inference_fns={
                    "Classifier_INCEPTION": partial(
                        inception_classification_full,
                        model=model,
                    )
                },
                batch_size=batch_size,
                verbose=verbose,
            )

            print(f"Finished inference for {dataset_name}")


    print("\nFinished parsing all UEA datasets.")
