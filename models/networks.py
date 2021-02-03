import torch.nn as nn
import torchvision.models as models


def get_resnet():
    # Create body of network
    model = models.resnet34(pretrained=True)
    children = list(model.children())[:6]
    return nn.Sequential(*children)


def uv(size, u_max=1, u_min=-1, v_max=1, v_min=-1):
    uv_grid = torch.FloatTensor([[[[u_min, u_max],
                                   [u_min, u_max]],
                                  [[v_max, v_max],
                                   [v_min, v_min]]]])

    return nn.functional.interpolate(uv_grid,
                                     size=[size, size],
                                     mode='bilinear',
                                     align_corners=True)[0]


class ReverseShuffle(nn.Module):
    # Add Layer of Spatia Mapping
    def __init__(self):
        super(ReverseShuffle, self).__init__()

    def forward(self, tensor):
        new_tensors = []
        batch, channel, res, res_a = tensor.shape
        grid_a = torch.arange(res).cuda()
        grid_b = grid_a + 1
        grid_a = grid_a % 2 == 0
        grid_b = grid_b % 2 == 0

        grid_b_expand = grid_b.expand(res, res).float()
        grid_a_expand = grid_a.expand(res, res).float()
        select_a = grid_a_expand * grid_b_expand.transpose(0, 1)
        select_b = grid_a_expand * grid_a_expand.transpose(0, 1)
        select_c = grid_b_expand * grid_a_expand.transpose(0, 1)
        select_d = grid_b_expand * grid_b_expand.transpose(0, 1)

        for select in [select_a, select_b, select_c, select_d]:
            masked = torch.masked_select(tensor, select.bool(), out=None)
            masked = masked.reshape(-1, channel, res // 2, res // 2)
            new_tensors.append(masked)

        return torch.cat(new_tensors, dim=1)


def conv_block(ni, nf, kernel_size=3, cc=True, icnr=True):
    layers = []
    if cc == True:
        layers += [AddCoordConv()]
        ni = ni + 2
    conv = nn.Conv2d(ni, nf, kernel_size, padding=kernel_size // 2)
    if icnr:
        conv.icnr = True

    relu = nn.LeakyReLU(inplace=True)

    bn = nn.BatchNorm2d(nf)
    drop = nn.Dropout(0.0)
    layers += [conv, relu, bn, drop]
    return nn.Sequential(*layers)


class DownRes(nn.Module):
    # Add Layer of Spatia Mapping
    def __init__(self, ic, oc, cc=True, kernel_size=3):
        super(DownRes, self).__init__()
        self.kernel_size = kernel_size
        self.oc = oc
        self.conv = conv_block(ic,
                               oc // 4,
                               cc=cc,
                               kernel_size=kernel_size,
                               icnr=False)

        self.rev_shuff = ReverseShuffle()

    def forward(self, x):
        unsqueeze_x = x.unsqueeze(0)
        if self.kernel_size % 2 == 0:
            x = x[:, :, :-1, :-1]
        x = self.conv(x)
        x = self.rev_shuff(x)

        upres_x = nn.functional.interpolate(unsqueeze_x,
                                            size=[self.oc, x.shape[2], x.shape[3]],
                                            mode='trilinear',
                                            align_corners=True)[0]
        x = x + (upres_x * .2)
        return x


class AddCoordConv(nn.Module):
    # Add Layer of Spatia Mapping
    def __init__(self):
        super(AddCoordConv, self).__init__()

    def forward(self, tensor, u_max=1, u_min=-1, v_max=1, v_min=-1):
        bs = int(tensor.shape[0])
        res = int(tensor.shape[2])
        grid = uv(res,
                  u_max=u_max,
                  u_min=u_min,
                  v_max=v_max,
                  v_min=v_min)

        uv_grid = torch.FloatTensor(grid).unsqueeze(0).expand([bs, 2, res, res])
        if tensor.is_cuda:
            uv_grid = uv_grid.cuda()
        cat_list = [uv_grid, tensor]
        uv_coord_tensor = torch.cat(cat_list, dim=1)
        return uv_coord_tensor


class ResHead(nn.Module):
    # Fully connected head
    def __init__(self, outclasses):
        super(ResHead, self).__init__()
        self.classes = outclasses

        self.conv_a = DownRes(128,256)
        self.conv_b = DownRes(256,512)
        self.conv_c = DownRes(512,512)
        self.conv_d = DownRes(512,512)
        self.av_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512, 512)
        self.linearb = nn.Linear(512, outclasses)

    def forward(self, x):

        x = self.conv_a(x)
        x = self.conv_b(x)
        x = self.conv_c(x)
        x = self.conv_d(x)
        x = x.view(-1,512)
        x = self.linearb(x)
        return x


class CustomResnet(nn.Module):
    # Combine body and head
    def __init__(self, outclasses):
        super(CustomResnet, self).__init__()
        self.classes = outclasses
        self.resnet = get_resnet()
        self.reshead = ResHead(outclasses)

    def forward(self, x):
        result = self.reshead(self.resnet(x))
        return result.view(-1, self.classes)

    def set_freeze(self, x):
        # Allow unfreezing of layers from head to body of model
        for i, layer in enumerate(self.resnet):
            needs_grad = i > (len(self.resnet) - 1) - x
            print(f'Layer {i} : Grad:{needs_grad}')
            for param in layer.parameters():
                param.requires_grad = needs_grad

    def lr_groups(self):
        # Different learning rate groups, Not using center group
        return self.resnet[:6], self.resnet[6:], self.reshead