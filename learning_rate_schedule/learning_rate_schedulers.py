import torch

# this file provide the learning rate scheduler for the training tasks

def rate(step, model_size, factor, warmup):
    """
    this function implement the learning rate scheduler mentioned in the paper
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def get_lr_scheduler(optimizer, config):
    '''
    given a optimizer(create from torch) and a configuration string
    the config must be like this:
    {
        "lr_scheduler": "LambdaLR",
        "d_model": 512,
        "warmup": 4000
    }
    return the learning rate scheduler
    '''
    if config["lr_scheduler"] == "LambdaLR":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, config["d_model"], factor=1, warmup=config["warmup"]
            ),
        )
    else:
        raise ValueError("Invalid learning rate scheduler")
    return lr_scheduler