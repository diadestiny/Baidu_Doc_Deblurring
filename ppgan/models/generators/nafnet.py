import paddle.nn as nn
import paddle.nn.functional as F
import paddle
from .builder import GENERATORS

# from basicsr.models.archs.local_arch import Local_Base

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

@GENERATORS.register()
class NAFNet(nn.Layer):

    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[ 1, 1, 1, 28 ], dec_blk_nums=[1, 1, 1, 1]):
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


# class NAFNetLocal(Local_Base, NAFNet):
#     def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
#         Local_Base.__init__(self)
#         NAFNet.__init__(self, *args, **kwargs)

#         N, C, H, W = train_size
#         base_size = (int(H * 1.5), int(W * 1.5))

#         self.eval()
#         with torch.no_grad():
#             self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

# if __name__ == "__main__":
#     img_channel = 3
#     width = 64
#     enc_blks = [1, 1, 1, 28]
#     middle_blk_num = 1
#     dec_blks = [1, 1, 1, 1]
#     model = NAFNet(img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
#                    enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
#     paddle.summary(model, (1, 3, 384, 384))
