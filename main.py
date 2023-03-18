from utils.dataset import Dataset
from model.solver import Solver
from model.oral_3d import Oral3D
import argparse

arg_parser = argparse.ArgumentParser(description="Training settings for Oral-3D")
arg_parser.add_argument('--test_mode', action="store_true", 
                        help='different test option')
arg_parser.add_argument('--data_root', dest='data_root', type=str, default='./data/mat',
                        help='location of training/val/test data and split info')
arg_parser.add_argument('--device', dest='device', type=str, default='cuda:0',
                        help='device to run the code')
arg_parser.add_argument('--train_n', dest='train_n', type=int, default=20,
                        help='epoch nums for training')
arg_parser.add_argument('--val_n', dest='val_n', type=int, default=5,
                        help='epoch nums for validation')
arg_parser.add_argument('--save_n', dest='save_n', type=int, default=20,
                        help='epoch nums for saving model')
arg_parser.add_argument('--d_start_n', dest='d_start_n', type=int, default=5,
                        help='epoch nums to start training discriminator')
arg_parser.add_argument('--g_lr', dest='g_lr', type=float, default=0.001,
                        help='epoch nums for saving model')
arg_parser.add_argument('--d_lr', dest='d_lr', type=float, default=0.001,
                        help='epoch nums for saving model')
arg_parser.add_argument('--mode', dest='mode', type=str, default='test',
                        help='train/test model')
arg_parser.add_argument('--test_only', action='store_true',
                        help='whether loading test data only')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    dataset = Dataset(args.data_root, test_only=args.test_only)
    
    oral_3d_model = Oral3D(device=args.device)
    solver = Solver(dataset, oral_3d_model, MPR=True, args=args)

    if args.mode == 'train':
         solver.train()
    else:
        solver.test()