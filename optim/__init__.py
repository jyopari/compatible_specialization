from torch.optim   import AdamW, SGD
from optim.adamw   import AdamW as AdamWN

def get_optimizer(net, args):
    # only train the router
    if args.router_method is not None:
        for name, param in net.named_parameters():
            if "policy" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        args.train_mlp = False # just to be sure

    # only train the mlp
    if args.train_mlp:
        for name, param in net.named_parameters():
            if "mlp" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    
    if args.optim == "adamw":
        optim = AdamW(net.parameters(), lr=args.lr, betas=(args.beta, args.beta2), weight_decay=args.wd)
    elif args.optim == "sgd":
        optim = SGD(net.parameters(), lr=args.lr, momentum=args.beta, weight_decay=args.wd)
    elif args.optim == "adamwn":
        optim = AdamWN(net.parameters(), lr=args.lr, betas=(args.beta, args.beta2), weight_decay=args.wd,normalize=True)
    return optim
