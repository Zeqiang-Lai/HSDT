
import torch
import torch.optim as optim
import models

import os
import argparse

from os.path import join
from utility import *
from utility.ssim import SSIMLoss
import torch.fft as fft

from tqdm import tqdm
from qqdm import qqdm, format_str

BAR = partial(qqdm)
# BAR = partial(tqdm, dynamic_ncols=True)

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1 / len(self.losses)] * len(self.losses)

    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            l = loss(predict, target)
            total_loss += l * weight
        return total_loss

    def extra_repr(self):
        return 'weight={}'.format(self.weight)


class FFTLoss(nn.Module):
    def __init__(self, rate=0.0):
        super().__init__()
        self.rate = rate

    def forward(self, predict, target):
        assert predict.shape == target.shape
        p_fft = fft.fftn(predict, dim=(-1, -2))
        t_fft = fft.fftn(target, dim=(-1, -2))
        p_fft = fft.fftshift(p_fft, dim=(-1, -2))
        t_fft = fft.fftshift(t_fft, dim=(-1, -2))
        p_fft = p_fft * self.mask(p_fft, self.rate)
        t_fft = t_fft * self.mask(t_fft, self.rate)
        return torch.mean(torch.pow(torch.abs(p_fft - t_fft), 2))

    @staticmethod
    def mask(img, rate):
        mask = torch.ones_like(img)
        rows, cols = img.shape[-2], img.shape[-1]
        mask[:, :, :, int(rows / 2 - rows * rate):int(rows / 2 + rows * rate), int(cols / 2 - cols * rate):int(cols / 2 + cols * rate)] = 0
        return mask


class FocalFrequencyLoss(nn.Module):
    """ Paper: Focal Frequency Loss for Image Reconstruction and Synthesis 
        Expect input'shape to be [..., W, H]
    """

    def __init__(self, alpha=1, norm='ortho'):
        super().__init__()
        self.alpha = alpha
        self.norm = norm

    def forward(self, output, target):
        o = fft.fftn(output, dim=(-1, -2), norm=self.norm)
        t = fft.fftn(target, dim=(-1, -2), norm=self.norm)
        d = torch.norm(torch.view_as_real(o - t), p=2, dim=-1)
        w = d.pow(self.alpha)
        w = (w - torch.amin(w, dim=(-1, -2), keepdim=True)) / torch.amax(w, dim=(-1, -2), keepdim=True)
        return torch.mean(w * d.pow(2))


def train_options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args
    parser.add_argument('--prefix', '-p', type=str, default='denoise',
                        help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
                        help='model architecture: ' +
                        ' | '.join(model_names))
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='name of the experiment, if not specified, arch will be used.')
    parser.add_argument('--batchSize', '-b', type=int,
                        default=16, help='training batch size. default=16')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate. default=1e-3.')
    parser.add_argument('--wd', type=float, default=0,
                        help='weight decay. default=0')
    parser.add_argument('--loss', type=str, default='l2',
                        help='which loss to choose.', choices=['l1', 'l2', 'smooth_l1', 'ssim', 'l2_ssim', 'l2_fft', 'fft'])
    parser.add_argument('--init', type=str, default='kn',
                        help='which init scheme to choose.', choices=['kn', 'ku', 'xn', 'xu', 'edsr'])
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true',
                        help='disable logger?')
    parser.add_argument('--threads', type=int, default=8,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed to use. default=2018')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                        help='not resume optimizer')
    parser.add_argument('--chop', action='store_true',
                        help='forward chop')
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--dataroot', '-d', type=str,
                        default='/home/wzliu/projects/data/ICVL64_31_100.db', help='data root')
    parser.add_argument('--clip', type=float, default=1e6)
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')
    parser.add_argument("-ed", type=str, default='', help="eval dataset name")
    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    return opt


def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    # dataset.length -= 1000
    # dataset.length = size or dataset.length

    """Split patches dataset into training, validation parts"""
    dataset = TransformDataset(dataset, common_transform)

    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)

    return train_loader


class Logger:
    def __init__(self, path):
        self.path = path

    def log(self, content):
        with open(self.path, 'a') as f:
            f.write(content + '\n')


class Engine(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None

        self.__setup()

    def __setup(self):
        self.basedir = join('checkpoints', self.opt.arch if self.opt.name is None else self.opt.name)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iteration = 0

        cuda = not self.opt.no_cuda
        self.device = 'cuda' if cuda else 'cpu'
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        from torchlight.utils import instantiate
        self.net = instantiate(self.opt.arch)
        # self.net.use_2dconv=True
        # self.net.bandwise=False
        # initialize parameters

        init_params(self.net, init_type=self.opt.init)  # disable for default initialization

        if len(self.opt.gpu_ids) > 1:
            from models.sync_batchnorm import DataParallelWithCallback
            self.net = DataParallelWithCallback(self.net, device_ids=self.opt.gpu_ids)

        if self.opt.loss == 'l2':
            self.criterion = nn.MSELoss()
        if self.opt.loss == 'l1':
            self.criterion = nn.L1Loss()
        if self.opt.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        if self.opt.loss == 'ssim':
            self.criterion = SSIMLoss(data_range=1, channel=31)
        if self.opt.loss == 'l2_ssim':
            self.criterion = MultipleLoss([nn.MSELoss(), SSIMLoss(data_range=1, channel=31)], weight=[1, 2.5e-3])
        if self.opt.loss == 'l2_fft':
            self.criterion = MultipleLoss([nn.MSELoss(), FocalFrequencyLoss()], weight=[1, 1])
        if self.opt.loss == 'fft':
            self.criterion = FFTLoss(rate=0.02)

        print(self.criterion)

        if cuda:
            self.net.to(self.device)
            self.criterion = self.criterion.to(self.device)

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs'), self.opt.prefix)
        self.logger = Logger(os.path.join(os.path.join(self.basedir, 'train.txt')))

        """Optimization Setup"""
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)

        total = sum([param.nelement() for param in self.net.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath, not self.opt.no_ropt)
        else:
            print('==> Building model..')
            print(self.net)

    def forward(self, inputs):
        if self.opt.chop:
            output = self.forward_chop(inputs)
        else:
            output = self.net(inputs)

        return output

    def forward_chop(self, x, base=16):
        n, c, b, h, w = x.size()
        h_half, w_half = h // 2, w // 2

        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)

        inputs = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.net(input_i) for input_i in inputs]

        output = torch.zeros_like(x)
        output_w = torch.zeros_like(x)

        output[..., 0:h_half, 0:w_half] += outputs[0][..., 0:h_half, 0:w_half]
        output_w[..., 0:h_half, 0:w_half] += 1
        output[..., 0:h_half, w_half:w] += outputs[1][..., 0:h_half, (w_size - w + w_half):w_size]
        output_w[..., 0:h_half, w_half:w] += 1
        output[..., h_half:h, 0:w_half] += outputs[2][..., (h_size - h + h_half):h_size, 0:w_half]
        output_w[..., h_half:h, 0:w_half] += 1
        output[..., h_half:h, w_half:w] += outputs[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        output_w[..., h_half:h, w_half:w] += 1

        output /= output_w

        return output

    def __step(self, train, inputs, targets):
        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        total_norm = None
        if self.get_net().bandwise:
            O = []
            for time, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                o = self.net(i)
                O.append(o)
                loss = self.criterion(o, t)
                if train:
                    loss.backward()
                loss_data += loss.item()
            outputs = torch.cat(O, dim=1)
        else:
            outputs = self.net(inputs)
            # outputs = torch.clamp(self.net(inputs), 0, 1)
            # loss = self.criterion(outputs, targets)

            # if outputs.ndimension() == 5:
            #     loss = self.criterion(outputs[:,0,...], torch.clamp(targets[:,0,...], 0, 1))
            # else:
            #     loss = self.criterion(outputs, torch.clamp(targets, 0, 1))
            if isinstance(outputs, list):
                gamma = 0.8
                loss = 0
                for i, pred in enumerate(outputs):
                    weight = gamma ** (len(outputs) - i - 1)
                    loss += weight * self.criterion(pred, targets)
                outputs = outputs[-1]
            else:
                loss = self.criterion(outputs, targets)

            if train:
                loss.backward()
            loss_data += loss.item()
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()

        return outputs, loss_data, total_norm

    def load(self, resumePath=None, load_opt=True):
        model_best_path = join(self.basedir, self.prefix, 'model_latest.pth')
        if os.path.exists(model_best_path):
            best_model = torch.load(model_best_path)

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath or model_best_path)
        # comment when using memnet
        self.epoch = checkpoint['epoch']
        self.iteration = checkpoint['iteration']
        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("LR: %f" % self.optimizer.param_groups[0]['lr'])
        ####
        self.get_net().load_state_dict(checkpoint['net'])

    def train(self, train_loader, warm_up=False):
        # print('\nEpoch: %d' % self.epoch)
        self.net.train()
        train_loss = 0
        total_psnr = 0
        best_avg_psnr = 0

        if warm_up:
            from utility.warmup_lr.scheduler import GradualWarmupScheduler
            scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=len(train_loader))

        pbar = BAR(total=len(train_loader))
        pbar.set_description('Epoch: {}'.format(format_str('yellow', self.epoch)))
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if not self.opt.no_cuda:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs, loss_data, total_norm = self.__step(True, inputs, targets)
            psnr = np.mean(cal_bwpsnr(outputs, targets))
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx + 1)

            total_psnr += psnr
            avg_psnr = total_psnr / (batch_idx + 1)

            if not self.opt.no_log:
                self.writer.add_scalar(
                    join(self.prefix, 'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(
                    join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)

            self.iteration += 1

            stat = {
                'AvgLoss': f'{avg_loss:.4e}',
                'Norm': f'{total_norm:.4e}',
                'PSNR': f'{avg_psnr:.4f}',
            }
            pbar.set_postfix(stat)
            pbar.update()

            if batch_idx % 20 == 0:
                self.logger.log('Epoch %d - [%d|%d]: AvgLoss: %.4e | Loss: %.4e | Norm: %.4e | PSNR: %.4f'
                                % (self.epoch, batch_idx, len(train_loader), avg_loss, loss_data, total_norm, avg_psnr))
            if avg_psnr > best_avg_psnr:
                best_avg_psnr = avg_psnr

            if warm_up:
                scheduler_warmup.step()

        self.epoch += 1
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)

    def validate(self, valid_loader, name):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_sam = 0

        pbar = BAR(total=len(valid_loader), dynamic_ncols=True)

        print('[i] Eval dataset {}...'.format(name))
        self.logger.log('[i] Eval dataset {}...'.format(name))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, loss_data, _ = self.__step(False, inputs, targets)
                psnr, ssim, sam = MSIQA(outputs, targets)

                validate_loss += loss_data
                avg_loss = validate_loss / (batch_idx + 1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx + 1)
                total_ssim += ssim
                avg_ssim = total_ssim / (batch_idx + 1)
                total_sam += sam
                avg_sam = total_sam / (batch_idx + 1)

                self.logger.log('Epoch %d | Loss: %.4e | PSNR: %.4f | SSIM: %.4f | SAM: %.4f'
                                % (self.epoch, avg_loss, avg_psnr, avg_ssim, avg_sam))

                stat = {
                    'Loss': f'{avg_loss:.4e}',
                    'PSNR': f'{avg_psnr:.4f}',
                    'SSIM': f'{avg_ssim:.4f}',
                    'SAM': f'{avg_sam:.4f}',
                }
                pbar.set_postfix(stat)
                pbar.update()


        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)

        return avg_psnr, avg_loss

    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iteration))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
        }

        state.update(kwargs)

        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    # saving result into disk
    def test_develop(self, test_loader, savedir=None, verbose=True):
        from scipy.io import savemat
        from os.path import basename, exists

        def torch2numpy(hsi):
            if self.net.use_2dconv:
                R_hsi = hsi.data[0].cpu().numpy().transpose((1, 2, 0))
            else:
                R_hsi = hsi.data[0].cpu().numpy()[0, ...].transpose((1, 2, 0))
            return R_hsi

        self.net.eval()
        test_loss = 0
        total_psnr = 0
        dataset = test_loader.dataset.dataset

        res_arr = np.zeros((len(test_loader), 3))
        input_arr = np.zeros((len(test_loader), 3))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs, loss_data, _ = self.__step(False, inputs, targets)

                test_loss += loss_data
                avg_loss = test_loss / (batch_idx + 1)

                res_arr[batch_idx, :] = MSIQA(outputs, targets)
                input_arr[batch_idx, :] = MSIQA(inputs, targets)

                """Visualization"""
                # Visualize3D(inputs.data[0].cpu().numpy())
                # Visualize3D(outputs.data[0].cpu().numpy())

                psnr = res_arr[batch_idx, 0]
                ssim = res_arr[batch_idx, 1]
                if verbose:
                    print(batch_idx, psnr, ssim)

                if savedir:
                    filedir = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0])
                    outpath = join(filedir, '{}.mat'.format(self.opt.arch))

                    if not exists(filedir):
                        os.mkdir(filedir)

                    if not exists(outpath):
                        savemat(outpath, {'R_hsi': torch2numpy(outputs)})

        return res_arr, input_arr

    def test_real(self, test_loader, savedir=None):
        """Warning: this code is not compatible with bandwise flag"""
        from scipy.io import savemat
        from os.path import basename
        self.net.eval()
        dataset = test_loader.dataset.dataset

        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs = inputs.cuda()

                outputs = self.forward(inputs)

                """Visualization"""
                input_np = inputs[0].cpu().numpy()
                output_np = outputs[0].cpu().numpy()

                display = np.concatenate([input_np, output_np], axis=-1)

                from torchvision.utils import save_image
                save_image(outputs[:, 0, 107, :, :], join('log_img', '{}.png'.format(batch_idx)))

                # Visualize3D(display)
                # Visualize3D(outputs[0].cpu().numpy())
                # Visualize3D((outputs-inputs).data[0].cpu().numpy())

                if savedir:
                    R_hsi = outputs.data[0].cpu().numpy()[0, ...].transpose((1, 2, 0))
                    savepath = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0], self.opt.arch + '.mat')
                    savemat(savepath, {'R_hsi': R_hsi})

        return outputs

    def get_net(self):
        if len(self.opt.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net
