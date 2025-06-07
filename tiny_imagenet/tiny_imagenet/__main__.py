import argparse as ap


def run_pretrain(*_, **__):
    from tiny_imagenet import pretrain
    pretrain.pretrain()


def run_finetune(*_, **__):
    from tiny_imagenet import finetune
    finetune.finetune()


if __name__ == '__main__':
    """Very simple command line interface that takes in some command and runs 
    the corresponding function.
    """
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)


    pretrain = subparsers.add_parser("pretrain")
    pretrain.set_defaults(func=run_pretrain)

    pretrain = subparsers.add_parser("finetune")
    pretrain.set_defaults(func=run_finetune)

    args = parser.parse_args()
    args.func(args)
