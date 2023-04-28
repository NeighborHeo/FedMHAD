import argparse

def init_args(server=True):
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start Flower server or client with experiment key.")
    
    # common arguments
    parser.add_argument("--experiment_key", type=str, default="test_key", required=False, help="Experiment key") 
    parser.add_argument("--model_name", type=str, default="vit_tiny_patch16_224", required=False, help="Model to use. Default: vit_tiny_patch16_224")
    parser.add_argument("--use_cuda", action="store_true", default=False, help="Set to true to use cuda. Default: False") 
    parser.add_argument("--port", type=int, default=8080, required=False, help="Port to use for the server. Default: 8080")
    parser.add_argument("--seed", type=int, default=1, required=False, help="seed")
    parser.add_argument("--toy", type=bool, default=False, required=False, help="Set to true to use only 10 datasamples for validation. Useful for testing purposes. Default: False" )
    parser.add_argument("--pretrained", type=bool, default=True, required=False, help="Set to true to use pretrained model. Default: False")
    # dataset arguments
    parser.add_argument("--num_clients", type=int, default=5, required=False, help="Number of clients to use. Default: 5")
    parser.add_argument("--dataset", type=str, default="pascal_voc", required=False, help="Dataset to use. Default: pascal_voc")
    parser.add_argument("--datapath", type=str, default="~/.data/", required=False, help="dataset path")
    parser.add_argument("--alpha", type=float, default=1.0, required=False, help="alpha")
    parser.add_argument("--noisy", type=float, default=0.0, required=False, help="Percentage of noisy data. Default: 0.0")
    # learning arguments
    parser.add_argument("--num_rounds", type=int, default=30, required=False, help="Number of rounds to run. Default: 30")
    parser.add_argument("--local_epochs", type=int, default=2, required=False, help="Number of local epochs. Default: 2")
    parser.add_argument("--learning_rate", type=float, default=0.00002, required=False, help="Learning rate. Default: 0.00002")
    parser.add_argument("--momentum", type=float, default=0.9, required=False, help="Momentum. Default: 0.9")
    parser.add_argument("--weight_decay", type=float, default=1e-5, required=False, help="Weight decay. Default: 1e-5")
    parser.add_argument("--batch_size", type=int, default=32, required=False, help="Batch size. Default: 32")
    # server arguments
    if server:
        parser.add_argument("--strategy", type=str, default="fedmhad", required=False, help="Strategy to use. Default: fedmhad")
    # client arguments
    if not server:
        parser.add_argument("--index", type=int, default=0, required=False, help="Index of the client")
        parser.add_argument("--dry", type=bool, default=False, required=False, help="Set to true to use only 10 datasamples for validation. Useful for testing purposes. Default: False" )

    args = parser.parse_args()
    if args.dataset == "pascal_voc":
        args.num_classes = 20
        args.task = "multilabel"
    elif args.dataset == "cifar10":
        args.num_classes = 10
        args.task = "singlelabel"
    return args  