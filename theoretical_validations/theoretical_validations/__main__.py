import argparse as ap


def run_tests(*_, **__):
    from theoretical_validations.run_tests import run_tests
    run_tests()


if __name__ == '__main__':
    """Very simple command line interface that takes in some command and runs 
    the corresponding function.
    """
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    pretrain = subparsers.add_parser("run_tests")
    pretrain.set_defaults(func=run_tests)

    args = parser.parse_args()
    args.func(args)
