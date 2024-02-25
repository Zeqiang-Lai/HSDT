from .arch import HSDTQRNN

def hsdt_qrnn():
    net = HSDTQRNN(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net