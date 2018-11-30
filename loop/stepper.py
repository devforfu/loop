import torch


class BaseStepper:
    """
    A thin wrapper that is intended to encapsulate a single training step.

    The instance of the class accept (x, y) pair and returns metrics collected
    after gradients update. The basic implementation doesn't include any
    additional properties and delegates attributes referencing to the
    underlying Loop class.
    """
    def __init__(self):
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

        return metrics

