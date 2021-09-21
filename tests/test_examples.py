from examples.feature_metrics import main as feature_metrics_examples
from examples.image_metrics import main as image_metrics_examples
from tests.test_gs import prepare_test


def test_image_metrics():
    prepare_test()
    image_metrics_examples()


def test_feature_metrics():
    prepare_test()
    feature_metrics_examples()
