import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def lr_loss_curve(lrs, losses, log_scale=True, zoom=None, ax=None, figsize=(10, 8)):
    """Plots curve reflecting the dependency between learning rate and training loss.

    Convenient to use with training.find_lr function.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
    if zoom is not None:
        min_lr, max_lr = zoom
        lrs, losses = zip(*[(x, y) for x, y in zip(lrs, losses) if min_lr <= x <= max_lr])
    ax.plot(lrs, losses)
    if log_scale:
        ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%1.0e'))
