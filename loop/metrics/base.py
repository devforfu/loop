"""
Metric functions expected to be called on batches of data.

In general case, each function expects model output and target to compute some
performance metric.
"""

def accuracy(out, y_true):
    y_hat = out.argmax(dim=-1).view(y_true.size(0), -1)
    y_true = y_true.view(y_true.size(0), -1)
    match = y_hat == y_true
    return match.float().mean()
