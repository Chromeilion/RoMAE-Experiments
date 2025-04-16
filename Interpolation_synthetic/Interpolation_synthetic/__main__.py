import argparse as ap

def run_preprocess(*_, **__):
    from Interpolation_synthetic import preprocess
    preprocess.preprocess()


def run_pretrain(*_, **__):
    from Interpolation_synthetic import pretrain
    pretrain.pretrain()


def run_finetune(*_, **__):
    from Interpolation_synthetic import finetune
    finetune.finetune()


if __name__ == '__main__':
    """Very simple command line interface that takes in some command and runs 
    the corresponding function.
    """
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    preprocess = subparsers.add_parser("preprocess")
    preprocess.set_defaults(func=run_preprocess)

    pretrain = subparsers.add_parser("pretrain")
    pretrain.set_defaults(func=run_pretrain)

    pretrain = subparsers.add_parser("finetune")
    pretrain.set_defaults(func=run_finetune)

    args = parser.parse_args()
    args.func(args)
