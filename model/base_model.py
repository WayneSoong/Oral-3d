import numpy as np

from utils.basic import *
from utils.eval import *
from utils.ssim import *
import torch
from utils.local_io import save_nii
from utils.interpolation import interpolation
from tqdm import tqdm
from model.loss import *
from termcolor import colored


class BaseModel:
    def __init__(self, dataset, network, model_name, MPR, cuda_id=None):
        # if the Mode is MPR, we make this as two-step registration method
        self.MPR = MPR
        self.model_name = model_name + '_MPR' if MPR else model_name
        self.name = network.name
        self.dataset = dataset
        self.cuda_id = cuda_id

        self.val_n = 10
        self.save_n = 50
        self.network = network if cuda_id is None else network.cuda(cuda_id)
        self._initial()
        self.best_score = 0.0
        self.optimizers = {}
        self.ssim_funct = SSIM(cuda_id=cuda_id)

    def _initial(self):
        model_dir = check_dir(join_path('output', self.model_name))
        save_space = check_dir(join_path(model_dir, self.name))
        self.log_dir = check_dir(join_path(save_space, 'log'))
        self.ckpt_dir = check_dir(join_path(save_space, 'ckpt'))
        self.result_dir = check_dir(join_path(save_space, 'result'))

    def _load_ckpt(self, epoch=0, BEST=False):
        # load checkpoint, get the parameter for keys in addition
        model_path = join_path(self.ckpt_dir, 'model_best.pth.tar') if BEST else join_path(self.ckpt_dir, 'model_%d.pth.tar' % epoch)
        state = torch.load(model_path)
        # load model param
        self.network.load_state_dict(state['param'])
        # load optimizer param
        for key in self.optimizers.keys():
            optim_state = state[key]
            self.optimizers[key].load_state_dict(optim_state)

    def _save_ckpt(self, epoch=0, BEST=False):
        checkpoint_path = join_path(self.ckpt_dir, 'model_best.pth.tar') if BEST else join_path(self.ckpt_dir, 'model_%d.pth.tar' % epoch)
        state = {'param': self.network.state_dict()}
        for optim_name, optimizer in self.optimizers.items():
            state[optim_name] = optimizer.state_dict()
        torch.save(state, checkpoint_path)

    def _adjust_learning_rate(self, optimizer, epoch_id, raw_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        if epoch_id < 150:
            lr = raw_lr * (0.1 ** ((epoch_id - 1) // 50))
        else:
            lr = raw_lr * (0.1 ** 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def inference(self, px_tensor, y=None, prior_shapes=None):

        # if the mode is only inference
        if y is None:
            generations = self.network.generate(px_tensor, VAL=True)
            generations_cpu = generations if self.cuda_id is None else generations.data.cpu().numpy()
            # if the shape is provided, registrant the image back to the old space
            if prior_shapes:
                generation_list = []
                for generation_cpu, prior_shape in zip(generations_cpu, prior_shapes):
                    generation_cpu[0:15, :, :] = -1
                    generation_cpu[65:, :, :] = -1
                    generation_list.append(interpolation(generation_cpu, prior_shape))
                generations_cpu = generation_list
            return generations_cpu

        # if the mode is training
        else:
            generations = self.network.generate(px_tensor, VAL=False)
            loss = proj_loss(generations, y) + rec_loss(generations, y)

            return loss, generations

        # return evaluation result
    def val(self, epoch_id, sampler, SAVE=False, mode='Val', EVAL=True):
        psnr_list = []
        ssim_list = []
        dice_list = []
        # relative_iou_list = []
        for batch_id in range(sampler.batch_n):
            # sample batch
            batch = sampler.get_batch()
            batch_img = batch['Ideal_PX']
            batch_size = len(batch_img)
            batch_shape = batch['PriorShape'] if self.MPR else None
            input_tensor = torch.tensor(np.array(batch_img), dtype=torch.float).cuda(self.cuda_id)
            generations = self.inference(input_tensor, prior_shapes=batch_shape)

            for item_id in range(batch_size):
                generation = generations[item_id]
                if EVAL:
                    # evaluate performance
                    CBCT = batch['CBCT'][item_id]
                    Bone = batch['Bone'][item_id]
                    MPR = batch['MPR'][item_id]
                    batch_shape = batch['PriorShape']

                    psnr_list.append(get_psnr(generation, CBCT))
                    ssim_list.append(self.ssim_funct.eval_ssim(generation, CBCT))
                    dice_list.append(get_dice(generation > -0.8, interpolation(MPR, batch_shape[item_id]) > -0.5))

                # save results
                if SAVE:
                    case_id = batch['Case_ID'][item_id]
                    if mode == 'Val':
                        generation_dir = check_dir(join_path(self.result_dir, mode + '_%d' % epoch_id))
                    elif mode == 'Test':
                        generation_dir = check_dir(join_path(self.result_dir, mode + '_%d' % self.dataset.TEST_MODE))
                    else:
                        raise ValueError('Unknown validation mode: %s' % mode)
                    generation_img = np.array((generation + 1) * 2000, dtype=np.uint16)
                    save_nii(generation_img, np.eye(4), 'case_%03d.nii.gz' % case_id, generation_dir)
        if EVAL:
            # calculate average value
            PSNR = np.mean(psnr_list)
            SSIM = np.mean(ssim_list)
            Dice = np.mean(dice_list)
            PSNR_std = np.std(psnr_list)
            SSIM_std = np.std(ssim_list)
            Dice_std = np.std(dice_list)


            avg_score = (PSNR / 20 + SSIM + Dice) / 3 * 100
            title = mode + '_%d' % epoch_id
            desc = f'{title}|PSNR: {PSNR:.4f} {PSNR_std:.4f}, SSIM: {SSIM:.4f} {SSIM_std:.4f}, ' \
                   f'Dice: {Dice:.4f} {Dice_std:.4f}, AVG: {avg_score:.2f}'
            print(colored(desc, 'blue'))
            return avg_score

    # test simulated case
    def test(self):
        self._load_ckpt(BEST=True)
        if self.dataset.TEST_MODE == 1:
            ssim_loss = self.val(epoch_id=0, sampler=self.dataset.test_sampler, SAVE=True, mode='Test', EVAL=True)
        elif self.dataset.TEST_MODE == 2 or self.dataset.TEST_MODE == 3:
            self.val(epoch_id=0, sampler=self.dataset.test_sampler, SAVE=True, mode='Test', EVAL=False)
        else:
            print('Unknown Test Mode')