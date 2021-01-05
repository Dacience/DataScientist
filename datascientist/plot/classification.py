import seaborn
import numpy

from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_moon
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification


def _make_blobs(*, n_samples=100, n_features=2):
    """For more info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html#sklearn.datasets.make_blobs
    """

    return make_blobs(n_samples=100, n_features=2)

def _make_gaussian_quantiles(*, mean=None, cov=1.0, n_samples=100, n_features=2, n_classes=3, shuffle=True, random_state=None):
    """For more info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_gaussian_quantiles.html#sklearn.datasets.make_gaussian_quantiles
    """    
        
    return make_gaussian_quantiles(mean=None, cov=1.0, n_samples=100, n_features=2, n_classes=3, shuffle=True, random_state=Nonea)
    
def _make_moon(*, n_samples=100, *, shuffle=True, noise=None, random_state=None):
    """For more info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html#sklearn.datasets.make_moons
    """ 

    return make_moon(n_samples=100, *, shuffle=True, noise=None, random_state=None)

def _make_circles(*, n_samples=100, *, shuffle=True, noise=None, random_state=None, factor=0.8):
    """For more info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html#sklearn.datasets.make_circles
    """

    return make_circles(n_samples=100, *, shuffle=True, noise=None, random_state=None, factor=0.8)

def _make_classification(*, n_samples=100, n_features=20, *, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, 
        n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True,
        random_state=None):
    """For more info visit :
        https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html#sklearn.datasets.make_classification
    """
    return make_classification(n_samples=100, n_features=20, *, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2,
    n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True,
    random_state=None)

def _plots(n_samples):
    _make_blobs(*, n_samples=100, n_features=2)
    _make_gaussian_quantiles(*, mean=None, cov=1.0, n_samples=100, n_features=2, n_classes=3, shuffle=True, random_state=None)
    _make_moon(*, n_samples=100, *, shuffle=True, noise=None, random_state=None)
    _make_circles(*, n_samples=100, *, shuffle=True, noise=None, random_state=None, factor=0.8)
    make_classification(n_samples=100, n_features=20, *, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)