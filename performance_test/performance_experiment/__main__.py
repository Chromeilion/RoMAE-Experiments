import argparse as ap

def run_experiment(*_, **__):
    from performance_experiment import run_exp
    run_exp.run()


if __name__ == '__main__':
    """Very simple command line interface that takes in some command and runs 
    the corresponding function.
    """
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    preprocess = subparsers.add_parser("run")
    preprocess.set_defaults(func=run_experiment)

    args = parser.parse_args()
    args.func(args)
