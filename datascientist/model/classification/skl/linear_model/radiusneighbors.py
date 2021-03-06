from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score

from sklearn.neighbors import RadiusNeighborsClassifier
import numpy as np


def _RadiusNeighborsClassifier(*, train, test, x_predict=None, metrics, radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', outlier_label=None, metric_params=None, n_jobs=None, **kwargs):
    """
    For more info visit :
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier
    """

    model = RadiusNeighborsClassifier(radius=radius, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p,
metric=metric, outlier_label=outlier_label, metric_params=metric_params, n_jobs=n_jobs, **kwargs)
    model.fit(train[0], train[1])
    model_name = 'Radius Neighbors Classifier'
    y_hat = model.predict(test[0])

    if metrics == 'accuracy':
        accuracy = accuracy_score(test[1], y_hat)

    if metrics == 'f1':
        accuracy = f1_score(test[1], y_hat)

    if metrics == 'jaccard':
        accuracy = jaccard_score(test[1], y_hat)

    if x_predict is None:
        return (model_name, accuracy, None)

    y_predict = model.predict(x_predict)
    return (model_name, accuracy, y_predict)
