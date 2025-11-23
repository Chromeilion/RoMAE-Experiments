import argparse as ap


def run_tests(*_, **__):
    from theoretical_validations.run_tests import run_tests
    run_tests()

def run_plot(*_, **__):
    from theoretical_validations.plot import plot
    plot()

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    pretrain = subparsers.add_parser("run_tests")
    pretrain.set_defaults(func=run_tests)

    pretrain = subparsers.add_parser("plot")
    pretrain.set_defaults(func=run_plot)

    args = parser.parse_args()
    args.func(args)
