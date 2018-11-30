def accuracy(logits, y_true):
    y_hat = logits.argmax(dim=-1).view(y_true.size(0), -1)
    y_true = y_true.view(y_true.size(0), -1)
    match = y_hat == y_true
    return match.float().mean()
