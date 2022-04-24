import argparse
import torch

def parse_args(mode="train"):
    ## 각종 파라미터 세팅
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')


    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Movielens dataset location')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
   parser.add_argument('--min_uc', type=int, default=5,
                        help='minimum user interaction number')
    parser.add_argument('--min_sc', type=int, default=0,
                        help='minimum user interaction number')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    args = parser.parse_args([])

    # Set the random seed manually for reproductibility.
    torch.manual_seed(args.seed)

    #만약 GPU가 사용가능한 환경이라면 GPU를 사용
    if torch.cuda.is_available():
        args.cuda = True
    device = torch.device("cuda" if args.cuda else "cpu")

    args = parser.parse_args()
    args.device = device
    print(args.device)

    return args
