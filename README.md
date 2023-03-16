# HSDT

Official PyTorch Implementation of [Hybrid Spectral Denoising Transformer with Learnable Query]().

*Zeqiang Lai, [Ying Fu](https://ying-fu.github.io/)*.

<img src="asset/arch.png" width="600px"/> 

🌟 **Hightlights**

- Superior hybrid spectral denoising transformer (HSDT), powered by a novel 3D guided spectral self-attention (GSSA), 3D spectral-spatial seperable convolution (S3Conv), and self-modulated feed-forward network (SM-FFN).
- Super fast convergence, 1 epoch to reach 39.5 PSNR on ICVL Gaussian 50, 3 epoch surpasses QRNN3D.
- Super lightweight, HSDT-S achieves comparable performance with SOTA with only 0.13M parameters, and HSDT-M outperforms SOTA by a large margin with only 0.52M parameters.

## Usage

Download the pretrained model at [Github Release]().

- Training, testing, and visualize results with [HSIR](https://github.com/bit-isp/HSIR).

```shell
python -m hsirun.test -a hsdt.hsdt -r ckpt/man_gaussian.pth -t icvl_512_30 icvl_512_50
python -m hsirun.train -a hsdt.hsdt -s schedule.gaussian
python -m hsirun.train -a hsdt.hsdt -s schedule.complex -r checkpoints/hsdts.hsdt/model_latest.pth
python -m hsiboard.app --logdir results
```

- Using our model.

```python
import torch
from hsdt import hsdt

net = hsdt()
x = torch.randn(4,1,31,64,64)
y = net(x)
```

- Using our components.

```python
import torch
from hsdt import (
    S3Conv
)

x = torch.randn(4,16,31,64,64)
block = S3Conv(16, 16, 3, 1, 1)
out = block(x) # [4,16,31,64,64]
```

## Performance

<details>
<summary>Gaussian denoising</summary>
<img src="asset/gaussian.png" width="800px"/> 
</details>

<details>
<summary>Complex denoising</summary>
<img src="asset/complex.png" width="800px"/> 

</details>


## Citation

```bibtex
@article{lai2022hsdt,
    title={Hybrid Spectral Denoising Transformer with Learnable Query}, 
    author={Zeqiang Lai, Ying Fu},
    year={2022},
}
```
