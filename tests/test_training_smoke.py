from sim.datasets import digits_loaders, set_seed
from sim.local_layers import DenseStochasticMLP
from sim.train_backprop import train_classifier


def test_digits_training_smoke():
    set_seed(0)
    train_loader, test_loader = digits_loaders(batch_size=256, seed=0)
    model = DenseStochasticMLP(64, hidden=(16,))
    result = train_classifier(model, train_loader, test_loader, epochs=1, lr=1e-3)
    assert len(result.train_loss) == 1
    assert 0.0 <= result.test_accuracy[-1] <= 1.0

