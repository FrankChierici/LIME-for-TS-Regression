import os
import yaml
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from data_loading import import_data
from explainability import get_heatmap
from xcm_model import xcm

def xcm_start(X_train, y_train, X_test, y_test, yticklabels):

    # Load dataset
    (
        X_train,
        y_train,
        X_test,
        y_test,
        y_train_nonencoded,
        y_test_nonencoded,
        yticklabels
    ) = import_data(X_train, y_train, X_test, y_test, yticklabels)
    print("X_train.shape: ", X_train.shape)
    print("X_test.shape: ", X_test.shape)

    # Instantiate the cross validator
    tskf = TimeSeriesSplit(
        n_splits=5,
    )

    # Instantiate the result dataframes
    train_val_epochs_accuracies = pd.DataFrame(
        columns=["Fold", "Epoch", "Accuracy_Train", "Accuracy_Validation"]
    )
    results = pd.DataFrame(
        columns=[
            "Dataset",
            "Model_Name",
            "Batch_Size",
            "Window_Size",
            "Fold",
            "MAE_Train",
            "MAE_Validation",
            "MAE_Test",
        ]
    )

    # Loop through the indices the split() method returns
    for index, (train_indices, val_indices) in enumerate(
        tskf.split(X_train)
    ):
        print("\nTraining on fold " + str(index + 1))

        # Generate batches from indices
        xtrain, xval = X_train[train_indices], X_train[val_indices]
        ytrain, yval, ytrain_nonencoded, yval_nonencoded = (
            y_train[train_indices],
            y_train[val_indices],
            y_train_nonencoded[train_indices],
            y_train_nonencoded[val_indices],
        )
        # Train the model

        model = xcm(
            input_shape=X_train.shape[1:],
            output_dim=1, # Let's work ONLY with 1 output per sample
            window_size=1,
        )

        model.compile(
            optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
        )
        h = model.fit(
            xtrain,
            ytrain,
            epochs=100,
            batch_size=16,
            verbose=1,
            validation_data=(xval, yval),
        )

        # Calculate accuracies
        fold_epochs_accuracies = np.concatenate(
            (
                pd.DataFrame(np.repeat(index + 1, 100)),
                pd.DataFrame(range(1, 100 + 1)),
                pd.DataFrame(h.history["mean_absolute_error"]),
                pd.DataFrame(h.history["val_mean_absolute_error"]),
            ),
            axis=1,
        )

        acc_train = mean_squared_error(
            ytrain, model.predict(xtrain)
        )
        acc_val = mean_squared_error(
            yval, model.predict(xval)
        )
        acc_test = mean_squared_error(
            y_test_nonencoded, model.predict(X_test)
        )

        # Add fold results to the dedicated dataframe
        train_val_epochs_accuracies = pd.concat(
            [
                train_val_epochs_accuracies,
                pd.DataFrame(
                    fold_epochs_accuracies,
                    columns=["Fold", "Epoch", "Accuracy_Train", "Accuracy_Validation"],
                ),
            ],
            axis=0,
        )

    # Train the model on the full train set
    print("\nTraining on the full train set")

    model = xcm(
        input_shape=X_train.shape[1:],
        output_dim=1,
        window_size=1,
    )

    print(model.summary())
    model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"]
    )
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=16,
        verbose=1,
    )

    # Add result to the results dataframe
    mse_test = mean_squared_error(
        y_test_nonencoded, model.predict(X_test)
    )
    results["Mean_Squared_Error_Full_Train"] = mse_test
    print("Mean Squared Error Test: {0}".format(mse_test))
    
    print(results)

    print('Mean absolute error xcm', ': ', mean_absolute_error(y_test_nonencoded, model.predict(X_test)))


    # Example of a heatmap from Grad-CAM for the first MTS of the test set
    get_heatmap(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        yticklabels=yticklabels
    )

    return model
