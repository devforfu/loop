import torch


class BaseStepper:

    def __init__(self, metrics=None):
        self.metrics = metrics
        self.loop = None

    def set_loop(self, loop):
        self.loop = loop

    def step(self, x, y, **kwargs):
        raise NotImplementedError()

    def __getattr__(self, item):
        if item not in self.__dict__:
            return getattr(self.__dict__['loop'], item)
        raise AttributeError(item)


class SimpleStepper(BaseStepper):
    """
    A thin wrapper encapsulating the model, its optimizer, a scheduler, and a
    loss function into single object.

    The stepper instance is invoked during each training iteration and returns
    the loss on batch.
    """

    def step(self, x, y, grad: bool=True):
        """
        Performs a single training step.

        Args:
            x: Features tensor.
            y: Target tensor.
            grad: If False, then the gradient is not computed, and the model's
                parameters are not updated.

        Returns:
            loss: The loss value on batch.

        """
        metrics = {}
        self.model.train(grad)

        with torch.set_grad_enabled(grad):
            out = self.model(x)
            loss = self.loss(out, y)
            metrics['loss'] = loss.item()

            if self.metrics is not None:
                for metric in self.metrics:
                    metrics[metric.__name__] = metric(out.cpu(), y.cpu())

            if grad:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.schedule.step()

        return metrics

