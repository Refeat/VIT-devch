import torch


def get_optimizer(model, cfg):
    """
    Return torch optimizer

    Args:
        model: Model you want to train
        cfg: Dictionary of optimizer configuration

    Returns:
        optimizer
    """
    optim_name = cfg["name"].lower()
    learning_rate = cfg["learning-rate"]
    args = cfg["args"]
    if optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, **args)
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, **args)
    elif optim_name == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, **args)
    else:
        raise NotImplementedError
    return optimizer


def get_criterion(cfg):
    """
    Return torch criterion

    Args:
        cfg: Dictionary of criterion configuration

    Returns:
        criterion
    """
    criterion_name = cfg["name"].lower()
    if criterion_name == "crossentropyloss":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    return criterion
