import importlib
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from golemai.config import LOGGER_LEVEL
from golemai.ml.metrics_ops import (
    false_negative,
    false_positive,
    true_negative,
    true_positive,
)
from sklearn.metrics import accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.multioutput import MultiOutputClassifier

logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_LEVEL)

SCORERS = {
    "auc": make_scorer(
        lambda y_true, y_pred: roc_auc_score(
            y_true, y_pred, average="weighted", multi_class="ovr"
        ),
        needs_proba=True,
    ),
    "accuracy": make_scorer(
        lambda y_true, y_pred: accuracy_score(y_true, y_pred)
    ),
    "tp": make_scorer(true_positive),
    "fp": make_scorer(false_positive),
    "fn": make_scorer(false_negative),
    "tn": make_scorer(true_negative),
}


def get_model_class(module: str, class_name: str) -> Any:

    try:

        module = importlib.import_module(module)
        ModelClass = getattr(module, class_name)

    except Exception as e:
        raise ValueError(f"Model init failed: {e}")
    else:
        return ModelClass


def get_classifier(
    module: str,
    class_name: str,
    unique_classes: int,
    multioutput: bool = False,
    model_params: Dict[str, Any] = None,
) -> Any:
    """
    Get a classifier model based on the provided module and class name.

    Args:
        module (str): The module containing the model class.
        class_name (str): The name of the model class.
        unique_classes (int): The number of unique classes in the target column.
        multioutput (bool): Whether to use a multioutput classifier.
        model_params (Dict[str, Any]): Parameters to pass to the model.

    Returns:
        Any: The instantiated model.
    """

    ModelClass = get_model_class(module=module, class_name=class_name)

    model = ModelClass(
        **model_params, num_class=unique_classes if unique_classes > 2 else 1
    )

    if multioutput:
        model = MultiOutputClassifier(model)

    return model


def evaluate_classifier(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    metrics: List[str],
    dataset_prefix: str = None,
) -> Dict[str, Any]:
    """
    Evaluate a classifier model.

    Args:
        model (Any): The model to evaluate.
        x (np.ndarray): The features.
        y (np.ndarray): The target.
        metrics (Dict[str, Any]): The metrics to evaluate.

    Returns:
        Dict[str, Any]: The evaluation results.
    """

    logger.info(
        f"evaluate_classifier: {model = } {x.shape = } "
        f"{y.shape = } {metrics = }"
    )

    results = {}
    eval_metrics = {metric: SCORERS[metric] for metric in metrics}

    for metric_name, metric_func in eval_metrics.items():

        metric_value = metric_func(model, x, y)
        results[
            (
                f"{dataset_prefix}_{metric_name}"
                if dataset_prefix is not None
                else metric_name
            )
        ] = metric_value

    return results


def cross_validate_classifier(
    module: str,
    class_name: str,
    model_params: Dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    cv: int,
    fit_on_train: bool = False,
) -> Tuple[Dict[str, Any], Any]:
    """
    Cross-validate a classifier model.

    Args:
        module (str): The module containing the model class.
        class_name (str): The name of the model class.
        unique_classes (int): The number of unique classes in the target column.
        multioutput (bool): Whether to use a multioutput classifier.
        model_params (Dict[str, Any]): Parameters to pass to the model.
        x (np.ndarray): The features.
        y (np.ndarray): The target.
        cv (int): The number of cross-validation folds.

    Returns:
        Dict[str, Any]: The cross-validation scores.
        Any: The fitted model if fit_on_train is True, None otherwise.

    """

    logger.info(
        f"cross_validate_classifier: {module = } {class_name = } "
        f"{model_params = } {x.shape = } {y.shape = } {cv = }"
    )

    unique_classes = len(np.unique(y))
    multioutput = True if y.flatten().shape[0] > y.shape[0] else False
    logger.info(f"unique_classes: {unique_classes = } {multioutput = }")

    model = get_classifier(
        module=module,
        class_name=class_name,
        unique_classes=unique_classes,
        multioutput=multioutput,
        model_params=model_params,
    )

    logger.info(f"Model from {module}.{class_name} initialized successfully")

    cv_scores = cross_validate(
        model,
        x,
        y,
        cv=cv,
        scoring=SCORERS,
        return_train_score=True,
        n_jobs=-1,
    )

    logger.info(f"Cross-validation completed successfully")

    if fit_on_train:
        model.fit(x, y)
    else:
        model = None

    return cv_scores, model
