from torch.optim import Adam, AdamW
import adabound

def get_optimizer(model, args):
    if args.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.optimizer == "adamW":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.optimizer == 'adbound':
        optimizer = adabound.AdaBound(model.parameters(), lr=args.lr, final_lr=0.1)

    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()

    return optimizer
