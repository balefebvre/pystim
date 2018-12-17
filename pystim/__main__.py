import argparse

from pystim.test import generate as generate_test
from pystim.dg import generate as generate_dg
from pystim.fipwc import generate as generate_fipwc
from pystim.utils import list_stimuli, configure, initialize, reinitialize


def print_list_stimuli(args):

    _ = args

    stimuli = list_stimuli()
    if len(stimuli) > 0:
        lines = ["    - {}".format(s) for s in stimuli]
        message = '\n'.join(lines)
        print(message)

    return


def defaults(args):

    _ = args

    raise NotImplementedError  # TODO replace.


def main():

    parser = argparse.ArgumentParser(description="A Python 3.6 module to generate visual stimuli.")
    parser.set_defaults(func=lambda args_: parser.print_usage())
    subparsers = parser.add_subparsers(title='positional arguments')

    subparser_config = subparsers.add_parser('config', help='config help')
    subparser_config.set_defaults(func=configure)

    subparser_init = subparsers.add_parser('init', help='init help')
    subparser_init.set_defaults(func=initialize)

    subparser_reinit = subparsers.add_parser('reinit', help='reinit help')
    subparser_reinit.set_defaults(func=reinitialize)

    subparser_list = subparsers.add_parser('list', help='list help')
    subparser_list.set_defaults(func=print_list_stimuli)

    subparser_generate = subparsers.add_parser('generate', help="generate help")
    subparser_generate.set_defaults(func=lambda args_: subparser_generate.print_usage())
    subparsers_generate = subparser_generate.add_subparsers(title="positional arguments")

    # Test.
    subparser_generate_test = subparsers_generate.add_parser('test', help='test help')
    subparser_generate_test.set_defaults(func=generate_test)

    # Drifting gratings.
    subparser_generate_dg = subparsers_generate.add_parser('dg', help='drifting gratings help')
    subparser_generate_dg.set_defaults(func=generate_dg)

    # Flashed images perturbed with checkerboard.
    subparser_generate_fipwc = subparsers_generate.add_parser('fipwc', help="flashed images perturbed with checkerboards")
    subparser_generate_fipwc.set_defaults(func=generate_fipwc)

    subparser_check = subparsers.add_parser('check', help="check help")
    subparser_check.set_defaults(func=defaults)

    args = parser.parse_args()
    args.func(args)
    # parser.print_usage()

    return


if __name__ == '__main__':

    main()
