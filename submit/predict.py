import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import glob
import json
import cv2
import numpy as np
import paddle
from PIL import Image
import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

patch_size = 1024
overlap = 0

class AvgPool2d(nn.Layer):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]
        self.train_size = train_size

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            train_size = self.train_size
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])
        if self.kernel_size[0] >= x.shape[-2] and self.kernel_size[1] >= x.shape[-1]:
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(axis=-1).cumsum(axis=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(axis=-1).cumsum_(axis=-2)
            s = nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            # print(x.shape, self.kernel_size)
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            ## compound module, go inside it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2D):
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m._output_size == 1
            setattr(model, n, pool)

class Local_Base():
    def convert(self, *args, train_size, **kwargs):
        replace_layers(self, *args, train_size=train_size, **kwargs)
        imgs = paddle.rand(train_size)
        with paddle.no_grad():
            self.forward(imgs)


def tensor2img(input_image, min_max=(-1., 1.), image_num=1, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor): the input image tensor array
        image_num (int): the convert iamge numbers
        imtype (type): the desired type of the converted numpy array
    """
    def processing(img, transpose=True):
        """"processing one numpy image.

        Parameters:
            im (tensor): the input image numpy array
        """
        # grayscale to RGB
        if img.shape[0] == 1:
            img = np.tile(img, (3, 1, 1))
        img = img.clip(min_max[0], min_max[1])
        img = (img - min_max[0]) / (min_max[1] - min_max[0])
        if imtype == np.uint8:
            # scaling
            img = img * 255.0
        # tranpose
        img = np.transpose(img, (1, 2, 0)) if transpose else img
        return img

    if not isinstance(input_image, np.ndarray):
        # convert it into a numpy array
        image_numpy = input_image.numpy()
        ndim = image_numpy.ndim
        if ndim == 4:
            image_numpy = image_numpy[0:image_num]
        elif ndim == 3:
            # NOTE for eval mode, need add dim
            image_numpy = np.expand_dims(image_numpy, 0)
            image_num = 1
        else:
            raise ValueError(
                "Image numpy ndim is {} not 3 or 4, Please check data".format(
                    ndim))

        if image_num == 1:
            # for one image, log HWC image
            image_numpy = processing(image_numpy[0])
        else:
            # for more image, log NCHW image
            image_numpy = np.stack(
                [processing(im, transpose=False) for im in image_numpy])

    else:
        # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = image_numpy.round()
    return image_numpy.astype(imtype)

class LayerNormFunction(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.shape
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        # y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        y = paddle.reshape(weight, [1, C, 1, 1]) * y + paddle.reshape(bias, [1, C, 1, 1])
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        # N, C, H, W = grad_output.size()
        N, C, H, W = grad_output.shape
        y, var, weight = ctx.saved_tensor()
        g = grad_output * paddle.reshape(weight, [1, C, 1, 1])
        mean_g = g.mean(axis=1, keepdim=True)

        mean_gy = (g * y).mean(axis=1, keepdim=True)
        gx = 1. / paddle.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        # return gx, (grad_output * y).sum(axis=3).sum(axis=2).sum(axis=0), grad_output.sum(axis=3).sum(axis=2).sum(axis=0), None
        return gx, (grad_output * y).sum(axis=3).sum(axis=2).sum(axis=0), grad_output.sum(axis=3).sum(axis=2).sum(
            axis=0)


class LayerNorm2d(nn.Layer):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = self.create_parameter(shape=[channels])
        self.add_parameter("weight", self.weight)
        self.bias = self.create_parameter(shape=[channels])
        self.add_parameter("bias", self.bias)
        # self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        # self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Layer):
    def forward(self, x):
        x1, x2 = x.chunk(2, axis=1)
        return x1 * x2


class NAFBlock(nn.Layer):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2D(c, dw_channel, 1, padding=0, stride=1, groups=1, bias_attr=True)
        self.conv2 = nn.Conv2D(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel,
                               bias_attr=True)
        self.conv3 = nn.Conv2D(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias_attr=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2D(1),
            nn.Conv2D(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias_attr=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2D(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias_attr=True)
        self.conv5 = nn.Conv2D(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias_attr=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = self.create_parameter(shape=[1, c, 1, 1])
        self.add_parameter("beta", self.beta)
        self.gamma = self.create_parameter(shape=[1, c, 1, 1])
        self.add_parameter("gamma", self.gamma)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Layer):

    def __init__(self, img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[ 1, 1, 1, 28 ], dec_blk_nums=[1, 1, 1, 1]):
        super().__init__()

        self.intro = nn.Conv2D(img_channel, width, 3, padding=1, stride=1, groups=1, bias_attr=True)
        self.ending = nn.Conv2D(width, img_channel, 3, padding=1, stride=1, groups=1, bias_attr=True)
        self.encoders = nn.LayerList()
        self.decoders = nn.LayerList()
        self.middle_blks = nn.LayerList()
        self.ups = nn.LayerList()
        self.downs = nn.LayerList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2D(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2D(chan, chan * 2, 1, bias_attr=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.shape
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 3, 512, 512), fast_imp=True, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with paddle.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

def process(src_image_dir, save_dir):
    model = NAFNetLocal()
    paddle.set_device('gpu:0')
    param_dict = paddle.load('./iter_196000_weight.pdparams')
    # print(param_dict['generator'].keys())
    model.load_dict(param_dict['generator'])
    model.eval()
    image_paths = glob.glob(os.path.join(src_image_dir, "*.png"))
    with paddle.no_grad():
        for image_path in image_paths:
            # do something
            lr_img = cv2.imread(image_path)
            h, w = lr_img.shape[:2]
            lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
            # resize
            # lr_img = paddle.vision.transforms.resize(lr_img, (512, 512), interpolation='bilinear')
            # lr_img = paddle.vision.transforms.functional.to_tensor(lr_img)
            # out = model(paddle.unsqueeze(lr_img, axis=0))
            # out_img = out[0][0]

            lr_img = paddle.vision.transforms.functional.to_tensor(lr_img)
            # paddle.set_printoptions(8, 100, 100,sci_mode = True)
            num_patch = 0
            out_img = paddle.zeros_like(lr_img)
            w1 = list(np.arange(0, w - patch_size, patch_size - overlap, dtype=np.int))
            h1 = list(np.arange(0, h - patch_size, patch_size - overlap, dtype=np.int))
            w1.append(w - patch_size)
            h1.append(h - patch_size)
            # import time
            # paddle.device.cuda.synchronize()
            # start = time.time()
            for i in h1:
                for j in w1:
                    num_patch += 1
                    lr_patch = lr_img[:,i:i + patch_size, j:j + patch_size]
                    # print(i,j)
                    out = model(paddle.unsqueeze(lr_patch,axis=0))
                    out_tensor = out[0][0]
                    # print(out)
                    # print(out_tensor.shape)
                    out_img[:,i:i + patch_size, j:j + patch_size] = out_tensor
            im = Image.fromarray(tensor2img(out_img,min_max=(0., 1.)))
            # im = paddle.vision.transforms.resize(im, (h, w), interpolation='bilinear')
            im.save(os.path.join(save_dir, os.path.basename(image_path)))
            # out_img
            # paddle.device.cuda.synchronize()
            # end = time.time()
            # total_time = end - start
            # print('total_time:{:.2f}'.format(total_time))
            # break

if __name__ == "__main__":
    assert len(sys.argv) == 3
    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]
    # src_image_dir = "../Datasets/deblur_testA/blur_image"
    # save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    process(src_image_dir, save_dir)