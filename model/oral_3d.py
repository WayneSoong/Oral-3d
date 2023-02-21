from model.base_model import *
import torch


class GAN(torch.nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.name = '%s_%s' % (generator.name, discriminator.name)
        self.generator = generator
        self.discriminator = discriminator

    def generate(self, input_tensor, VAL):
        generations = self.generator.generate(input_tensor, VAL)
        return generations.squeeze(1)


class Oral_3D(BaseModel):
    def __init__(self, dataset, network, model_name='Oral_3D', MPR=True, cuda_id=None):
        super().__init__(dataset, network, model_name, MPR, cuda_id)
        self.optimizers = {'g_optim': torch.optim.Adam(self.network.generator.parameters(), lr=0.001),
                           'd_optim': torch.optim.SGD(self.network.discriminator.parameters(), lr=0.01)}
        self.network.discriminator.cuda_id = cuda_id

    def train(self, start_epoch=0, epoch_n=100):
        # load saved ckpt if start_epoch != 0
        if start_epoch:
            self._load_ckpt(start_epoch)

        sampler = self.dataset.train_sampler

        self.best_score = self.val(start_epoch, self.dataset.val_sampler)
        optim_g = self.optimizers['g_optim']
        optim_d = self.optimizers['d_optim']
        start_epoch += 1

        for epoch_id in range(start_epoch, start_epoch + epoch_n):
            optim_g = self._adjust_learning_rate(optim_g, epoch_id, 0.001)
            optim_d = self._adjust_learning_rate(optim_d, epoch_id, 0.001)
            with tqdm(total=sampler.data_n) as epoch_pbar:
                d_loss_list = []
                g_loss_list = []
                for batch_id in range(sampler.batch_n):
                    batch = sampler.get_batch()
                    input_tensor = torch.tensor(batch['Ideal_PX'], dtype=torch.float)
                    gt_tensor = torch.tensor(batch['MPR'], dtype=torch.float) if self.MPR \
                        else torch.tensor(batch['CBCT'], dtype=torch.float)
                    if self.cuda_id is not None:
                        input_tensor = input_tensor.cuda(self.cuda_id)
                        gt_tensor = gt_tensor.cuda(self.cuda_id)

                    # update generator
                    loss_g, generations_MPR = self.inference(input_tensor, y=gt_tensor)
                    g_loss_list.append(loss_g.data.cpu().numpy())
                    if epoch_id > 50:
                        # self.network.discriminator = self.network.discriminator.eval()
                        loss_g += self.network.discriminator.inference(generations_MPR) * 0.05

                    optim_g.zero_grad()
                    loss_g.backward()
                    optim_g.step()

                    # update discriminator
                    if epoch_id > 50:
                        # self.network.discriminator = self.network.discriminator.train()
                        loss_d = self.network.discriminator.discriminate(generations_MPR.detach(), gt_tensor)

                        optim_d.zero_grad()
                        loss_d.backward()
                        optim_d.step()
                        d_loss_list.append(loss_d.data.cpu().numpy())
                    else:
                        loss_d = 0.0
                        d_loss_list.append(loss_d)
                    # update
                    desc = f'Epoch:{epoch_id:04d}|loss_g {loss_g:.4f}, loss_d {loss_d:.4f}'
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(input_tensor.shape[0])
                desc = f'Epoch:{epoch_id:04d}|loss_g {np.mean(g_loss_list):.4f}, loss_d {np.mean(d_loss_list):.4f}'
                epoch_pbar.set_description(desc)
                epoch_pbar.close()

            # validation
            SAVE = epoch_id % self.save_n == 0
            if SAVE:
                self._save_ckpt(epoch_id, BEST=False)
            if epoch_id % self.val_n == 0:
                score = self.val(epoch_id, self.dataset.val_sampler, SAVE)
                if score > self.best_score:
                    self.best_score = score
                    self._save_ckpt(epoch_id, BEST=True)

