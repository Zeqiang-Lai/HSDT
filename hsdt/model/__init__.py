from .arch import HSDT


def hsdt():
    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net

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


""" ablations
"""


def hsdt_conv3d():
    from . import arch
    import torch.nn as nn
    arch.Conv3d = nn.Conv3d

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_pixelwise():
    from . import arch
    from .attention import PixelwiseTransformerBlock
    arch.TransformerBlock = PixelwiseTransformerBlock

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


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


def hsdt_gfn():
    from . import arch
    from .attention import GFNTransformerBlock
    arch.TransformerBlock = GFNTransformerBlock

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_ssa():
    from . import arch
    from .attention import SSATransformerBlock
    arch.TransformerBlock = SSATransformerBlock

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_sepconv():
    from . import arch
    import torch.nn as nn
    from .sepconv import SepConv3D_DW
    arch.Conv3d = SepConv3D_DW.of(nn.Conv3d)
    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_s3conv_sep():
    from . import arch
    import torch.nn as nn
    from .sepconv import SepConv3D_MSEP
    arch.Conv3d = SepConv3D_MSEP.of(nn.Conv3d)
    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_s3convs():
    from . import arch
    import torch.nn as nn
    from .sepconv import SepConv3D_MS
    arch.Conv3d = SepConv3D_MS.of(nn.Conv3d)
    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net

# ablation one by one


def hsdt_wossa():
    from . import arch
    from .attention import DummyTransformerBlock
    arch.TransformerBlock = DummyTransformerBlock
    arch.UseBN = False

    net = HSDT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net


def hsdt_wossa_conv3d():
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

# baseline [running]
# baseline + ssa
# baseline + gssa
# baseline + gssa + sm-ffn [done]
# basseline + gssa + sm-ffn + s3conv [done]
