import argparse
import torch

def parse_args(mode="train"):
    ## 각종 파라미터 세팅
    parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')


    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Movielens dataset location')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--n_heldout', type=int, default=3000,
                        help='number of heldout users')
    parser.add_argument('--min_uc', type=int, default=5,
                        help='minimum user interaction number')
    parser.add_argument('--min_sc', type=int, default=0,
                        help='minimum user interaction number')
    parser.add_argument('--model', type=str, default='multivae',
                        help='select model')
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
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
