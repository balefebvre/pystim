import argparse

from pystim.fipwc import generate as generate_fipwc


def check(args):

    _ = args

    raise NotImplementedError  # TODO replace.


def main():

    parser = argparse.ArgumentParser(description="A Python 3.6 module to generate visual stimuli.")
    parser.set_defaults(func=lambda args_: parser.print_usage())
    subparsers = parser.add_subparsers(title='positional arguments')

    subparser_generate = subparsers.add_parser('generate', help="generate help")
    subparser_generate.set_defaults(func=lambda args_: subparser_generate.print_usage())
    subparsers_generate = subparser_generate.add_subparsers(title="positional arguments")

    subparser_generate_fipwc = subparsers_generate.add_parser('fipwc', help="flashed images perturbed with checkerboards")
    subparser_generate_fipwc.set_defaults(func=generate_fipwc)

    subparser_check = subparsers.add_parser('check', help="check help")
    subparser_check.set_defaults(func=check)

    args = parser.parse_args()
    args.func(args)
    # parser.print_usage()

    return


if __name__ == '__main__':

    main()
