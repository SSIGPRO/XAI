import tensorflow.keras as keras
import numpy as np
import tensorflow as tf
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt

class Classifier_INCEPTION:
    def __init__(
        self,
        output_directory,
        input_shape,
        nb_classes,
        verbose=False,
        build=True,
        batch_size=64,
        lr=0.001,
        nb_filters=32,
        use_residual=True,
        use_bottleneck=True,
        depth=6,
        kernel_size=41,
        nb_epochs=750,
        bottleneck_size=32
    ):

        self.output_directory = output_directory
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.batch_size = batch_size
        self.bottleneck_size = bottleneck_size
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose

        if build:
            self.model = self.build_model(input_shape, nb_classes)
            if verbose:
                self.model.summary()

            # Keras 3 requires .weights.h5 for weights-only files
            self.model.save_weights(
                self.output_directory + "model_init.weights.h5"
            )

    # Get model
    def get_model(self):
        return self.model

    # Shuffle training data
    def shuffle(self, x, y):
        indices = np.random.permutation(len(x))
        return x[indices], y[indices]

    # Inception module
    def _inception_module(
        self,
        input_tensor,
        stride=1,
        activation="linear"
    ):

        # Bottleneck layer
        if (
            self.use_bottleneck
            and int(input_tensor.shape[-1]) > self.bottleneck_size
        ):

            input_inception = keras.layers.Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding="same",
                activation=activation,
                use_bias=False
            )(input_tensor)

        else:
            input_inception = input_tensor

        # Multi-scale kernel sizes
        kernel_size_s = [
            self.kernel_size // (2 ** i)
            for i in range(3)
        ]

        conv_list = []

        # Parallel convolution branches
        for kernel_size in kernel_size_s:
            conv = keras.layers.Conv1D(
                filters=self.nb_filters,
                kernel_size=kernel_size,
                strides=stride,
                padding="same",
                activation=activation,
                use_bias=False
            )(input_inception)

            conv_list.append(conv)

        # Max-pooling branch
        max_pool = keras.layers.MaxPool1D(
            pool_size=3,
            strides=stride,
            padding="same"
        )(input_tensor)

        conv_pool = keras.layers.Conv1D(
            filters=self.nb_filters,
            kernel_size=1,
            padding="same",
            activation=activation,
            use_bias=False
        )(max_pool)

        conv_list.append(conv_pool)

        # Concatenate branches
        x = keras.layers.Concatenate(axis=2)(conv_list)

        # Batch normalization
        x = keras.layers.BatchNormalization()(x)

        # ReLU activation
        x = keras.layers.Activation("relu")(x)
        return x

    # Residual shortcut
    def _shortcut_layer(self,input_tensor,out_tensor):
        shortcut = keras.layers.Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding="same",
            use_bias=False
        )(input_tensor)

        shortcut = keras.layers.BatchNormalization()(shortcut)
        x = keras.layers.Add()([shortcut,out_tensor])
        x = keras.layers.Activation("relu")(x)
        return x
    
    # Build InceptionTime model
    def build_model(self, input_shape,nb_classes):
        input_layer = keras.layers.Input(shape=input_shape)
        x = input_layer
        input_res = input_layer

        # Stack Inception modules
        for d in range(self.depth):
            x = self._inception_module(x)
            
            # Residual connection after every 3 modules
            if (
                self.use_residual
                and d % 3 == 2
            ):
                x = self._shortcut_layer(input_res,x)
                input_res = x

        # Global average pooling
        gap_layer = keras.layers.GlobalAveragePooling1D()(x)
        
        # Classification layer
        output_layer = keras.layers.Dense(
            nb_classes,
            activation="softmax"
        )(gap_layer)

        # Create model
        model = keras.models.Model(inputs=input_layer,outputs=output_layer)

        # Compile model
        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(
                learning_rate=self.lr
            ),
            metrics=["accuracy"]
        )
        return model

    # Train model
    def fit(
        self,
        x_train,
        y_train,
        x_validate,
        y_validate,
        x_test,
        y_test
    ):

        start_time = time.time()
        n = x_train.shape[0]

        # Training history
        accs = []
        losses = []
        val_accs = []
        val_losses = []

        # Keras 3 requires .weights.h5 for weights-only files
        file_path = (self.output_directory+ "best_model.weights.h5")

        # Best validation loss
        min_val_loss = np.inf

        # Learning-rate reduction
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=50,
            min_lr=0.0001
        )

        reduce_lr.set_model(self.model)
        reduce_lr.on_train_begin()
        reduce_lr.verbose = self.verbose

        # Training loop
        for e in range(self.nb_epochs):
            # Shuffle training data only
            x_train, y_train = self.shuffle(x_train, y_train)
            if self.verbose:
                print(f"\nEpoch {e + 1}/{self.nb_epochs}")
            epoch_loss = 0.0
            epoch_acc = 0.0
            denom = 0

            # Mini-batch training
            for i in range(
                self.batch_size,
                n + self.batch_size,
                self.batch_size
            ):

                max_i = min(n,i)
                cur_i = (i - self.batch_size)
                x = x_train[cur_i:max_i]
                y = y_train[cur_i:max_i]
                
                x = tf.convert_to_tensor(x, dtype=tf.float32)
                y = tf.convert_to_tensor(y, dtype=tf.float32)
                print("Batch after conversion:", type(x))

                # Train on batch
                curr_loss, curr_acc = (
                    self.model.train_on_batch(x,y))
                epoch_loss += curr_loss
                epoch_acc += curr_acc
                denom += 1

            # Average training metrics
            epoch_loss /= denom
            epoch_acc /= denom

            # Validation evaluation
            val_loss, val_acc = (
                self.model.evaluate(
                    x_validate,
                    y_validate,
                    batch_size=self.batch_size,
                    verbose=False
                )
            )

            # Store metrics
            losses.append(epoch_loss)
            accs.append(epoch_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Print results
            if self.verbose:
                print(
                    f"loss: {epoch_loss:.4f} - "
                    f"accuracy: {epoch_acc:.4f} - "
                    f"val_loss: {val_loss:.4f} - "
                    f"val_accuracy: {val_acc:.4f}"
                )

            # Save best model based on validation loss
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                self.model.save_weights(file_path)
                if self.verbose:
                    print("Saved new best model.")

            # Update learning rate
            reduce_lr.on_epoch_end(
                epoch=e,
                logs={
                    "loss": epoch_loss,
                    "val_loss": val_loss
                }
            )

        # Plot accuracy
        plt.figure()
        plt.ylim(top=1.0,bottom=0.0)
        plt.plot(accs,label="train")
        plt.plot(val_accs,label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="best")
        plt.savefig(self.output_directory+ "acc.pdf")
        plt.close()

        # Plot loss
        plt.figure()
        plt.plot(losses,label="train")
        plt.plot(val_losses,label="validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.savefig(self.output_directory+ "loss.pdf")
        plt.close()

        # Save training history
        df = pd.DataFrame(
            index=[
                i
                for i in range(
                    self.nb_epochs
                )
            ],
            columns=[
                "loss",
                "acc",
                "val_loss",
                "val_acc"
            ]
        )

        df["loss"] = losses
        df["acc"] = accs
        df["val_loss"] = val_losses
        df["val_acc"] = val_accs
        df.to_csv(self.output_directory+ "history.csv")

        # Training duration
        duration = (time.time()- start_time)
        
        # Load best validation model
        self.model.load_weights(file_path)

        # FINAL TEST EVALUATION
        # Test data is used only here.
        test_loss, test_acc = (
            self.model.evaluate(
                x_test,
                y_test,
                batch_size=self.batch_size,
                verbose=False
            )
        )
        # Generate test predictions
        # NumPy -> TensorFlow
        x_test_tf = tf.convert_to_tensor(x_test,dtype=tf.float32)

        # Keras model prediction
        # Output is a TensorFlow tensor
        y_pred_tf = self.model(x_test_tf, training=False)

        # TensorFlow -> NumPy -> PyTorch
        y_pred_torch = torch.from_numpy(y_pred_tf.numpy())

        # Convert predictions to class indices
        y_pred_classes = torch.argmax(y_pred_torch,dim=1)

        # Convert one-hot labels to PyTorch
        y_true_torch = torch.from_numpy(y_test)

        # Convert one-hot labels to class indices
        y_true_classes = torch.argmax(y_true_torch,dim=1)

        # Calculate final test accuracy
        final_accuracy = (
            (y_pred_classes == y_true_classes)
            .float()
            .mean()
            .item()
        )

        # Check the prediction type
        print("\nPrediction information:")
        print("TensorFlow prediction:", type(y_pred_tf))
        print("PyTorch prediction   :", type(y_pred_torch))
        print("Prediction shape     :", y_pred_torch.shape)
        print("Prediction dtype     :", y_pred_torch.dtype)

        # Save final test results
        results = pd.DataFrame({
            "test_loss": [test_loss],
            "test_accuracy": [test_acc],
            "calculated_accuracy": [
                final_accuracy
            ],
            "training_duration_seconds": [
                duration
            ]
        })

        results.to_csv(self.output_directory+ "df_metrics.csv",index=False)

        # Print final results
        print("\n" + "=" * 70)
        print("Final Test Results")
        print("=" * 70)
        print( f"Test loss     : {test_loss:.4f}")
        print(f"Test accuracy : {test_acc:.4f}")
        print(f"Accuracy      : {final_accuracy:.4f}")
        print(f"Training time : {duration:.2f} seconds")
        print("=" * 70)

        # Clear TensorFlow session
        keras.backend.clear_session()
        return results, y_pred_torch

    # Predict using saved weights
    def predict(
        self,
        x_test,
        y_true,
        return_df_metrics=True
    ):

        start_time = time.time()
        model_path = (self.output_directory+ "best_model.weights.h5")

        # Load best weights
        self.model.load_weights(model_path)
        
        # NumPy -> TensorFlow
        x_test_tf = tf.convert_to_tensor(x_test,dtype=tf.float32)

        # Keras prediction
        y_pred_tf = self.model(x_test_tf,training=False)
        
        # TensorFlow -> NumPy -> PyTorch
        y_pred_torch = torch.from_numpy(y_pred_tf.numpy())
        print("\nPrediction information:")
        print("TensorFlow prediction:", type(y_pred_tf))
        print("PyTorch prediction   :", type(y_pred_torch))
        print("Prediction shape     :", y_pred_torch.shape)
        print("Prediction dtype     :", y_pred_torch.dtype)

        if return_df_metrics:

            # PyTorch prediction -> class indices
            y_pred_classes = torch.argmax(y_pred_torch,dim=1)

            # Convert labels to PyTorch
            y_true_torch = torch.from_numpy(y_true)

            # Convert labels to class indices
            if y_true_torch.ndim > 1:
                y_true_classes = torch.argmax(y_true_torch,dim=1)
            else:
                y_true_classes = y_true_torch

            # Calculate accuracy
            accuracy = (
                (y_pred_classes == y_true_classes)
                .float()
                .mean()
                .item()
            )

            duration = time.time() - start_time
            results = pd.DataFrame({
                "accuracy": [accuracy],
                "prediction_time_seconds": [
                    duration
                ]
            })
            return results

        else:
        # Return PyTorch prediction
            return y_pred_torch