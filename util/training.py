import math
import numpy as np
import torch


############################################################################
#  Learning Rate
############################################################################


def set_lr_sched(epochs, iters, mult):
    """ Learning rate schedule, using cyclical learning rate

    Args:
      epochs (int): how many cycles to build
      iters (int): iter count for first epoch
      mult (float): how much longer each epoch will be

    Returns:
      list: float values representing learning rate during training
    """

    mult_iter = iters
    iter_stack = []
    for a in range(epochs):
        iter_stack += [math.cos((x / mult_iter) * 3.14) * .5 + .5 for x in (range(int(mult_iter)))]
        mult_iter *= mult
    return iter_stack


def set_opt_lr(opt, lr):
    """ Sets optimizer learning rate, using different values for each sub group

    Args:
      opt (torch.optimizer): optimizer to set learning rate on
      lr (np.array): array of learning rate values
    """

    opt.param_groups[0]['lr'] = lr[0]
    opt.param_groups[1]['lr'] = lr[1]
    opt.param_groups[2]['lr'] = lr[2]


############################################################################
#  Training loop
############################################################################    


def train(train_dict, x, y, lr, lr_lookup, current_iter):
    """ Train network for one iter

    Args:
      train_dict (dict): dictionary containing network, optimizer, loss and loader
      x (torch.tensor): input image from dataset
      y (torch.tensor): actual y value from dataset
      lr (float): base learning rate overall multiplier
      lr_lookup (list): float values representing learning rate during training
      current_iter (int): which iteration we are on, used to lookup learning rate

    Returns:
      torch.tensor: mse loss
      torch.tensor: point loss
    """

    lr_mult = lr_lookup[current_iter]
    set_opt_lr(train_dict["opt"], lr * lr_mult)

    train_dict["opt"].zero_grad()
    yhat = train_dict["net"](x)
    mse_loss = train_dict["mse_crit"](yhat, y.float())
    point_loss = train_dict["point_crit"](yhat, y.float())

    # Testing with no point loss, works better in some situations
    loss = mse_loss  # +  point_loss  #  might not help
    loss.backward()
    train_dict["opt"].step()
    return mse_loss, point_loss


def test(train_dict, x, y):
    """ Test network for one iter

    Args:
      train_dict (dict): dictionary containing network, optimizer, loss and loader
      x (torch.tensor): input image from dataset
      y (torch.tensor): actual y value from dataset

    Returns:
      torch.tensor: mse loss
      torch.tensor: point loss
    """

    yhat = train_dict["net"](x)
    mse_loss = train_dict["mse_crit"](yhat, y.float())
    point_loss = train_dict["point_crit"](yhat, y.float())
    return mse_loss, point_loss


def one_run(train_dict, freeze, lr_list, lr_array):
    """ Run through one cycle of cyclical learning

    Args:
      train_dict (dict): dictionary containing network, optimizer, loss and loader
      freeze (int): how many layers to freeze on the model
      lr_list (float): float values representing learning rate during training
      lr_array (np.array): array of learning rate values
    """

    current_iter = 0
    current_epoch = 0
    done = False
    train_dict["net"].set_freeze(freeze)

    while not done:
        epoch_test_mse_loss = []
        epoch_train_mse_loss = []
        epoch_test_point_loss = []
        epoch_train_point_loss = []

        # TRAIN LOOP
        train_dict["net"].train()
        for x, y in train_dict["train_loader"]:

            if current_iter > len(lr_list) - 1:
                done = True
                break

            x, y = x.cuda(), y.cuda()
            mse_loss, point_loss = train(train_dict,
                                         x, y,
                                         lr_array,
                                         lr_list,
                                         current_iter)

            epoch_train_mse_loss.append(torch.mean(mse_loss).cpu().detach().numpy())
            epoch_train_point_loss.append(torch.mean(point_loss.float()).cpu().detach().numpy())

            current_iter += 1

        # TEST LOOP
        train_dict["net"].eval()
        for x, y in train_dict["test_loader"]:
            x, y = x.cuda(), y.cuda()
            mse_loss, point_loss = test(train_dict,
                                        x, y)

            epoch_test_mse_loss.append(torch.mean(mse_loss).cpu().detach().numpy())
            epoch_test_point_loss.append(torch.mean(point_loss.float()).cpu().detach().numpy())

        print(
            f'train mse_loss: {np.array(epoch_train_mse_loss).mean()}'
            f'   train point_loss: {np.array(epoch_train_point_loss).mean()}')
        print(
            f'test mse_loss: {np.array(epoch_test_mse_loss).mean()}'
            f'   test point_loss: {np.array(epoch_test_point_loss).mean()}')

        current_epoch += 1
    print('Done')
