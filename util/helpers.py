import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


############################################################################
#  Export
############################################################################


def save_jit(output_name, run, model):
    """ Export trained model with JIT

    Args:
      output_name (string): output filename
      model (ResNet): neural net to generate predictions
      run (int): cycle count
    """

    model = model.eval()
    model.train(False)

    example = torch.rand(1, 3, 128, 128).cuda()
    traced_script_module = torch.jit.trace(model, example)

    fname = f'output/{output_name}_{run}'
    traced_script_module.save(f'{fname}.ptc')


############################################################################
#  Preview Functions
############################################################################


def lr_preview(lr_list):
    """ Draw learning rate schedule as graph

    Args:
      lr_list (list): list of learning rates to plat
    """

    fig = plt.figure(figsize=(4, 2))
    plt.plot(lr_list)
    plt.ylabel('Learning Rate Schedule')
    plt.show()
    plt.close(fig)


def bar_chart(plot, b):
    """ View bar chart representing raw data output from network

    Args:
      plot (matplotlib.pyplot.plt): plot to draw in
      b (torch.tensor): output from neural net
    """

    objects = [f'BS:{(a+1)}' for a in range(51)]
    y_pos = np.arange(len(objects))
    performance = b.squeeze().detach().cpu().numpy()

    plot.bar(y_pos,
             performance,
             align='center',
             alpha=0.9,
             color='lawngreen')


def conv(img):
    """ Conver color from BGR to RGB

    Args:
      img (np.array): image
    """

    return cv2.cvtColor(np.flip(img, 1), cv2.COLOR_BGR2RGB)


def view_dataset(train_data, transform):
    """ View data samples from dataset

    Args:
      transform (NormDenorm): image normalization object
      train_data (LandMarkGenerator): dataloader
    """

    fig, ax = plt.subplots(3, 2, figsize=(6, 9))
    fig.patch.set_alpha(0.0)

    # Set images
    count = 0
    for idx in range(3):
        a, b = next(iter(train_data))

        ax[count, 0].cla()
        bar_chart(ax[count, 0], b)

        ax[count, 1].cla()
        ax[count, 1].imshow(conv(transform.denorm(a)),
                            interpolation='lanczos')
        count += 1

    title_dict = {4: f"Blendshapes",
                  5: f"OpenCV Render",
                  }

    # Format
    count = 0
    for a in ax.flat:
        a.set_xticklabels('')
        a.set_xticks([])

        a.tick_params(axis="y",
                      labelcolor="white")
        a.set_facecolor('k')

        if count % 2 != 0:
            a.set_yticklabels('')
            a.set_yticks([])

        if count in title_dict.keys():

            a.text(0.5,
                   -0.15,
                   title_dict[count],
                   size=12,
                   ha="center",
                   transform=a.transAxes,
                   color='black')
        count += 1

    plt.show()


def view_predictions(res_net, train_data):
    """ View predicted results in matplotlib

    Args:
      res_net (ResNet): neural net to generate predictions
      train_data (LandMarkGenerator): dataloader
    """

    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    fig.patch.set_alpha(0.0)

    # Set images
    count = 0
    for idx in range(3):
        a, b = next(iter(train_data))
        yhat = res_net(a.cuda().unsqueeze(0))

        ax[count, 0].cla()
        bar_chart(ax[count, 0], b)

        ax[count, 1].cla()
        img = train_data.get_pred_face(b.unsqueeze(0).numpy()).astype(np.float32)
        ax[count, 1].imshow(conv(img), interpolation='lanczos')

        ax[count, 2].cla()
        bar_chart(ax[count, 2], yhat)

        ax[count, 3].cla()
        img = train_data.get_pred_face(yhat.cpu().detach().numpy()).astype(np.float32)
        ax[count, 3].imshow(conv(img), interpolation='lanczos')

        count += 1

    title_dict = {8: f"Blendshapes",
                  9: f"OpenCV Render",
                  10: f"Predicted Blendshapes",
                  11: f"Predicted OpenCV Render"
                  }

    # Format
    count = 0
    for a in ax.flat:
        a.set_xticklabels('')
        a.set_xticks([])

        a.tick_params(axis="y",
                      labelcolor="white")
        a.set_facecolor('k')

        if count % 2 != 0:
            a.set_yticklabels('')
            a.set_yticks([])

        if count in title_dict.keys():
            a.text(0.5, -0.15,
                   title_dict[count],
                   size=12, ha="center",
                   transform=a.transAxes,
                   color='black')
        count += 1

    plt.show()
