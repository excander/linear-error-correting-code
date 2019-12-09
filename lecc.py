import argparse
import src

parser = argparse.ArgumentParser(prog='lecc', description='Linear error-correcting code.')

subparsers = parser.add_subparsers(title='mode', description='Valid modes', help='additional help')

parser_gencode = subparsers.add_parser('generator', help='generating linear code and decode vector')
parser_gencode.add_argument('r', type=int, help='number of check bits')
parser_gencode.add_argument('n', type=int, help='length of a code word')
parser_gencode.add_argument('t', type=int, help='number of errors to correct')
parser_gencode.add_argument('--out-file', dest='out', type=str, default='code.data',
                            help='file to write data for coder/decoder (default: code.data)')
parser_gencode.set_defaults(func=src.generate)

parser_coder = subparsers.add_parser('encoder', help='encode message m and adding error vector -e.')
parser_coder.add_argument('inputfile', type=str, help='file with encoder data in pickle format')
parser_coder.add_argument('m', type=str, help='message')
parser_coder.add_argument('-e', dest='e', type=str, help='error', default=None)
parser_coder.set_defaults(func=src.encode)

parser_decoder = subparsers.add_parser('decoder', help='decode message and subtracting error vector')
parser_decoder.add_argument('inputfile', type=str, help='file with decoder data in pickle format')
parser_decoder.add_argument('y', type=str, help='message with error')
parser_decoder.set_defaults(func=src.decode)

args = parser.parse_args()
args.func(args)
