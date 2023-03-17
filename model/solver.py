import torch
import os
from munch import Munch
from utils.basic import *
import numpy as np
from utils.basic import *
from utils.eval import *
from utils.ssim import *
from utils.local_io import save_nii
from utils.interpolation import interpolation
from tqdm import tqdm
from model.loss import *
from termcolor import colored


class Solver:
    def __init__(self, dataset, model, MPR, args=None):
        self.dataset = dataset
        self.model = model
        
        # if the Mode is MPR, we make this as two-step registration method
        self.MPR = MPR
        self.model_name = model.name + '_MPR' if MPR else model.name
        self.dataset = dataset
        self.args = args
        self._initial_save_sapce()
        self.best_score = 0.0
        self.ssim_funct = SSIM(device=self.args.device)
        if 'cuda' in self.args.device:
            self.model.to(self.args.device)
        
    def train(self,):
        sampler = self.dataset.train_sampler

        # running in the first time of the model
        eval_scores = self.eval(sampler=self.dataset.val_sampler, get_eval_score=True, save_generations=False)
        self.print_eval_scores(eval_scores, title='Val_0')
        self.overall_score = eval_scores['overall_score']
        
        optim_g =  torch.optim.Adam(self.model.generator.parameters(), lr=self.args.g_lr)   # 0.001
        optim_d = torch.optim.SGD(self.model.discriminator.parameters(), lr=self.args.d_lr) # 0.01

        for epoch_id in range(1, self.args.train_n+1):
            
            optim_g = self._adjust_learning_rate(optim_g, epoch_id, self.args.g_lr)
            optim_d = self._adjust_learning_rate(optim_d, epoch_id, self.args.d_lr)
            
            with tqdm(total=sampler.data_n) as epoch_pbar:
                d_loss_list = []
                g_loss_list = []
                for batch_id in range(sampler.batch_n):
                    batch = sampler.get_batch()
                    px_tensor = torch.tensor(batch['Ideal_PX'], dtype=torch.float, device=self.args.device)
                    gt_tensor = torch.tensor(batch['MPR'], dtype=torch.float, device=self.args.device) if self.MPR \
                        else torch.tensor(batch['CBCT'], dtype=torch.float, device=self.args.device)

                    # update generator
                    generations = self.model.generate(px_tensor, VAL=False)
                    loss_g = proj_loss(generations, gt_tensor) + rec_loss(generations, gt_tensor)
                    g_loss_list.append(loss_g.item())
                    
                    if epoch_id > self.args.d_start_n:
                        loss_g += self.model.discriminator.inference(generations) * 0.05

                    optim_g.zero_grad()
                    loss_g.backward()
                    optim_g.step()

                    # update discriminator
                    if epoch_id > self.args.d_start_n:
                        loss_d = self.model.discriminator.discriminate(generations.detach(), gt_tensor)
                        optim_d.zero_grad()
                        loss_d.backward()
                        optim_d.step()
                        loss_d = loss_d.item()
                    else:
                        loss_d = 0.0
                    d_loss_list.append(loss_d)
                    
                    # update progree bar
                    desc = f'Epoch:{epoch_id:04d}|loss_g {loss_g:.4f}, loss_d {loss_d:.4f}'
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(px_tensor.shape[0])
                desc = f'Epoch:{epoch_id:04d}|loss_g {np.mean(g_loss_list):.4f}, loss_d {np.mean(d_loss_list):.4f}'
                epoch_pbar.set_description(desc)
                epoch_pbar.close()
            if epoch_id % self.args.save_n == 0:
                self._save_ckpt(epoch_id)
                
            if epoch_id % self.args.val_n == 0:
                eval_scores = self.eval(sampler=self.dataset.val_sampler, get_eval_score=True, 
                                        save_generations=False, save_dir='Val_%d'%epoch_id)
                self.print_eval_scores(eval_scores, title='Val_%d'%epoch_id)
                overall_score = eval_scores['overall_score']
                if overall_score > self.best_score:
                    self.best_score = overall_score
                    self._save_ckpt(epoch=0)

        # return evaluation result
    def eval(self, sampler, get_eval_score=True, save_generations=False, save_dir=None):
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
            px_tensor = torch.tensor(np.array(batch_img), dtype=torch.float, device=self.args.device)
            
            generations = self.model.generate(px_tensor, VAL=True)

            generations_cpu = generations.detach() if 'cpu' in self.args.device else generations.detach().cpu().numpy()
            # if the shape is provided, registrant the image back to the old space
            if batch_shape:
                generation_list = []
                for generation_cpu, prior_shape in zip(generations_cpu, batch_shape):
                    # ignore part of the generation slices
                    ignore_slice_n=15
                    generation_cpu[:ignore_slice_n, :, :] = -1
                    generation_cpu[-ignore_slice_n:, :, :] = -1
                    generation_list.append(interpolation(generation_cpu, prior_shape))
                generations_cpu = generation_list

            for item_id in range(batch_size):
                generation = generations_cpu[item_id]
                if get_eval_score:
                    # evaluate performance
                    CBCT = batch['CBCT'][item_id]
                    MPR = batch['MPR'][item_id]
                    batch_shape = batch['PriorShape']

                    psnr_list.append(get_psnr(generation, CBCT))
                    ssim_list.append(self.ssim_funct.eval_ssim(generation, CBCT))
                    dice_list.append(get_dice(generation > -0.8, interpolation(MPR, batch_shape[item_id]) > -0.5))

                # save results
                if save_generations:
                    generation_dir = join_path(self.result_dir, save_dir)
                    os.makedirs(generation_dir, exist_ok=True)
                    case_id = batch['Case_ID'][item_id]
                    generation_img = np.array((generation + 1) * 2000, dtype=np.uint16)
                    save_nii(generation_img, np.eye(4), 'case_%03d.nii.gz' % case_id, generation_dir)
                    
        if get_eval_score:
            psnr_mean, psnr_std = np.mean(psnr_list), np.std(psnr_list)
            ssim_mean, ssim_std = np.mean(ssim_list), np.std(ssim_list)
            dice_mean, dice_std = np.mean(dice_list), np.std(dice_list)
            overall_score = self.get_overall_score(psnr_mean, ssim_mean, dice_mean)
            return Munch(psnr_mean=psnr_mean, psnr_std=psnr_std,
                         ssim_mean=ssim_mean, ssim_std=ssim_std,
                         dice_mean=dice_mean, dice_std=dice_std,
                         overall_score=overall_score
                         )
        else:
            return None
    
        # test simulated case
    def test(self):
        self._load_ckpt(epoch=0)
        eval_scores = self.eval(sampler=self.dataset.test_sampler, get_eval_score=True, save_generations=True, save_dir='TEST')
        self.print_eval_scores(eval_scores, title='TEST')
    
    @staticmethod
    def print_eval_scores(eval_scores, title):
        psnr_mean, psnr_std = eval_scores['psnr_mean'], eval_scores['psnr_std']
        ssim_mean, ssim_std = eval_scores['ssim_mean'], eval_scores['ssim_std']
        dice_mean, dice_std = eval_scores['dice_mean'], eval_scores['dice_std']
        overall_score = eval_scores['overall_score']
        
        desc = f'{title}|PSNR: {psnr_mean:.4f} {psnr_std:.4f}, SSIM: {ssim_mean:.4f} {ssim_std:.4f}, Dice: {dice_mean:.4f} {dice_std:.4f}, AVG: {overall_score:.2f}'
        print(colored(desc, 'blue'))
    
    @staticmethod
    def get_overall_score(psnr, ssim, dice):
        return (psnr / 20 + ssim + dice) / 3 * 100
        
    def _initial_save_sapce(self):
        save_space = check_dir(join_path('output', self.model_name))
        self.log_dir = check_dir(join_path(save_space, 'log'))
        self.ckpt_dir = check_dir(join_path(save_space, 'ckpt'))
        self.result_dir = check_dir(join_path(save_space, 'result'))

    def _load_ckpt(self, epoch=0):
        # if epoch_id is 0, load the best model
        # if epoch_id is -1, load the latest model
        if epoch == 0:
            ckpt_name = 'model_best.pth.tar'
        else:
            ckpt_name = 'model_%d.pth.tar' % epoch
        
        # load checkpoint, get the parameter for keys in addition
        ckpt_path = join_path(self.ckpt_dir, ckpt_name)
        state = torch.load(ckpt_path)
        self.model.load_state_dict(state)


    def _save_ckpt(self, epoch=0):
        # if epoch_id is 0, save as the best model
        if epoch == 0:
            ckpt_name = 'model_best.pth.tar'
        else:
            ckpt_name = 'model_%d.pth.tar' % epoch
        ckpt_path = join_path(self.ckpt_dir, ckpt_name)
        torch.save(self.model.state_dict(), ckpt_path)
    
    def _adjust_learning_rate(self, optimizer, epoch_id, raw_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        if epoch_id <= 100:
            lr = raw_lr * (0.1 ** (epoch_id // 50))
        else:
            lr = raw_lr * (0.1 ** 2)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer