import onnx
from onnx import optimizer
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2


############################################################################
#  Export
############################################################################


def export_onnx(res_net, name):
    # Export network to ONNX and then optimize
    output_names = ["output1"]
    dummy_input = torch.zeros([1, 3, 96, 96]).cuda()
    res_net.eval()
    res_net.train(False)
    torch.onnx.export(res_net, dummy_input, name + ".onnx", verbose=True, output_names=output_names)

    model = onnx.load(name + '.onnx')
    passes = ["fuse_transpose_into_gemm", "fuse_bn_into_conv", "fuse_add_bias_into_conv"]
    optimized_model = optimizer.optimize(model, passes)
    onnx.save(optimized_model, name + '_opt.onnx')


############################################################################
#  Preview
############################################################################


def lr_preview(lr_list):
    fig = plt.figure(figsize=(4, 2))
    plt.plot(lr_list)
    plt.ylabel('Learning Rate Schedule')
    plt.show()
    plt.close(fig)


def bar_chart(plot, b):
    objects = [f'BS:{(a+1)}' for a in range(51)]
    y_pos = np.arange(len(objects))
    performance = b.squeeze().detach().cpu().numpy()
    plot.bar(y_pos, performance, align='center', alpha=0.9, color='lawngreen')


def conv(img):
    return cv2.cvtColor(np.flip(img, 1), cv2.COLOR_BGR2RGB)


def view_dataset(train_data, transform):
    fig, ax = plt.subplots(3, 2, figsize=(6, 9))
    fig.patch.set_alpha(0.0)

    # Set images
    count = 0
    for idx in range(3):
        a, b = next(iter(train_data))

        ax[count, 0].cla()
        bar_chart(ax[count, 0], b)

        ax[count, 1].cla()
        ax[count, 1].imshow(conv(transform.denorm(a)), interpolation='lanczos')
        count += 1

    title_dict = {4: f"Blendshapes",
                  5: f"OpenCV Render",
                  }

    # Format
    count = 0
    for a in ax.flat:
        a.set_xticklabels('')
        a.set_xticks([])

        a.tick_params(axis="y", labelcolor="white")
        a.set_facecolor('k')

        if count % 2 != 0:
            a.set_yticklabels('')
            a.set_yticks([])

        if count in title_dict.keys():
            a.text(0.5, -0.15, title_dict[count], size=12, ha="center", transform=a.transAxes, color='black')
        count += 1

    plt.show()


def view_predictions(res_net, train_data):
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
