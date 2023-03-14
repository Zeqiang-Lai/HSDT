from .arch import HSDT
from .attention import GSSA, SMFFN, TransformerBlock
from .sepconv import S3Conv

def hsdt():
    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_4():
    net = HSDT(1, 4, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_8():
    net = HSDT(1, 8, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_24():
    net = HSDT(1, 24, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_32():
    net = HSDT(1, 32, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_deep():
    net = HSDT(1, 16, 7, [1, 3, 5])
    net.use_2dconv = False
    net.bandwise = False
    return net


""" Extension
"""


def hsdt_pnp():
    net = HSDT(2, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_ssr():
    from .arch import HSDTSSR
    net = HSDTSSR(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


""" ablations
"""


def hsdt_pixelwise():
    from . import arch
    from .attention import PixelwiseTransformerBlock
    arch.TransformerBlock = PixelwiseTransformerBlock

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net

# ablation of ffn


def hsdt_ffn():
    from . import arch
    from .attention import FFNTransformerBlock
    arch.TransformerBlock = FFNTransformerBlock

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_ffn_flex():
    from . import arch
    from .attention import FFNTransformerBlock
    from functools import partial
    arch.TransformerBlock = partial(FFNTransformerBlock, flex=True)

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_gdfn():
    from . import arch
    from .attention import GDFNTransformerBlock
    arch.TransformerBlock = GDFNTransformerBlock

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_smffn1():
    from . import arch
    from .attention import GFNTransformerBlock
    arch.TransformerBlock = GFNTransformerBlock

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net

# ablation of ssa


def hsdt_ssa():
    from . import arch
    from .attention import SSATransformerBlock
    arch.TransformerBlock = SSATransformerBlock

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net

# ablation of s3conv


def hsdt_conv3d():
    from . import arch
    import torch.nn as nn
    arch.Conv3d = nn.Conv3d

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_s3conv_sep():
    from . import arch
    import torch.nn as nn
    from .sepconv import SepConv_DP
    arch.Conv3d = SepConv_DP.of(nn.Conv3d)
    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_s3conv_seq():
    from . import arch
    import torch.nn as nn
    from .sepconv import S3Conv_Seq
    arch.Conv3d = S3Conv_Seq.of(nn.Conv3d)
    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_s3conv1():
    from . import arch
    import torch.nn as nn
    from .sepconv import S3Conv1
    arch.Conv3d = S3Conv1.of(nn.Conv3d)
    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


""" Break down
"""


def baseline_s3conv():
    from . import arch
    from .attention import DummyTransformerBlock
    arch.TransformerBlock = DummyTransformerBlock
    arch.UseBN = False

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def baseline_conv3d():
    from . import arch
    import torch.nn as nn
    arch.Conv3d = nn.Conv3d
    from .attention import DummyTransformerBlock
    arch.TransformerBlock = DummyTransformerBlock
    arch.UseBN = False

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def baseline_gssa():
    from . import arch
    import torch.nn as nn
    arch.Conv3d = nn.Conv3d
    from .attention import FFNTransformerBlock
    arch.TransformerBlock = FFNTransformerBlock
    arch.UseBN = True

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def baseline_ssa():
    from . import arch
    import torch.nn as nn
    arch.Conv3d = nn.Conv3d
    from .attention import SSAFFNTransformerBlock
    arch.TransformerBlock = SSAFFNTransformerBlock
    arch.UseBN = True

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net
