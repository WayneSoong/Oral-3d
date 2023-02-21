from utils.dataset import Dataset
from model.encoder_MPR import Encoder_MPR
from model.discriminator import PatchDiscriminator
from model.oral_3d import GAN, Oral_3D
import argparse

arg_parser = argparse.ArgumentParser(description="Training settings for Oral-3D")
arg_parser.add_argument('--test_mode', action="store_true", help='whether to test existing model')
arg_parser.add_argument('--data_root', dest='data_root', type=str, default='./data/mat',
                        help='location of training/val/test data and split info')
arg_parser.add_argument('--cuda_id', dest='cuda_id', type=int, default=1,
                        help='cuda device to run the code')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    dataset = Dataset(args.data_root)
    gan_networks = GAN(generator=Encoder_MPR(name='Encoder_MPR_GLPL'), discriminator=PatchDiscriminator(cuda_id=args.cuda_id))
    model = Oral_3D(dataset, gan_networks, MPR=True, cuda_id=0)
    print('Running model %s with network %s' % (model.model_name, gan_networks.name))
    if args.test_mode:
        model.test()
    else:
        model.train(start_epoch=0, epoch_n=200)
        