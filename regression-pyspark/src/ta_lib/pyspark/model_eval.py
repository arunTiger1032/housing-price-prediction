"""Module for Model Evaluation and Interpretation."""


import seaborn as sns

from tigerml.pyspark.model_eval.model_eval import (
    ClassificationReport,
    PySparkReport,
    RegressionReport,
    distribution_plot,
    exp_var,
    generate_confusion_cell_col,
    get_binary_classification_metrics,
    get_binary_classification_plots,
    get_binary_classification_report,
    get_classification_scores,
    get_regression_metrics,
    get_regression_plots,
    get_regression_report,
    mape,
    plot_interaction,
    scatter_plot,
    stacked_bar_plot,
    wmape,
)

sns.set()

_VALID_REGRESSION_METRICS_ = {
    "Explained Variance": "exp_var",
    "RMSE": "rmse",
    "MAE": "mae",
    "MSE": "mse",
    "MAPE": "mape",
    "WMAPE": "wmape",
    "R.Sq": "r2",
}


# FIX ME
# Interactive y,yhat plot
# How can we get this in pyspark

# -----------------------------------------------------------------------
# Classification - Individual Model (WIP)
# -----------------------------------------------------------------------
_BINARY_CLASSIFICATION_METRICS_ = {
    "Accuracy": "accuracy",
    "F1 Score": "f1",
    "TPR": "tpr",
    "FPR": "fpr",
    "Precision": "precision",
    "Recall": "recall",
    "AuROC": "auROC",
    "AuPR": "auPR",
}
