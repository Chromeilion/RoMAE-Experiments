import argparse as ap


def run_train(*_, **__):
    from uea import train
    train.train()


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    pretrain = subparsers.add_parser("train")
    pretrain.set_defaults(func=run_train)

    args = parser.parse_args()
    args.func(args)
