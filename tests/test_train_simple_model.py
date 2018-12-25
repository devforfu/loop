from torch.optim import Adam
from torch.nn import functional as F

from loop import train_classifier
from loop.training import find_lr
from loop.torch_helpers.modules import TinyNet


def test_training_model_with_loop(mnist):
    model = TinyNet()
    opt = Adam(model.parameters(), lr=1e-2)

    result = train_classifier(model, opt, data=mnist, epochs=3, batch_size=512, num_workers=0)

    assert result['phases']['valid'].metrics['accuracy'][-1] > 0.95


def test_finding_optimal_lr(mnist):
    model = TinyNet()
    opt = Adam(model.parameters())

    losses = find_lr(model, opt, mnist[0], F.cross_entropy, batch_size=256)

    assert losses is not None
