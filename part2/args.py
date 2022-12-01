import argparse

parser = argparse.ArgumentParser(description='574 final project')

parser.add_argument(
    '--lr',
    type=float,
    default=0.003,
    help='Learning rate',
)

parser.add_argument(
    '--weight_init',
    default='random',
    choices=['random', 'zero'],
    help='Weight initialization mode',
    type=str
)

parser.add_argument(
    '--activation',
    default='sigmoid',
    choices=['sigmoid', 'relu', 'tanh'],
    help='Activation function',
    type=str
)

parser.add_argument(
    '--weight_decay',
    default=0.001,
    help='L2 weight regulariziation factor',
    type=float,
)

parser.add_argument(
    '--units1',
    default=12,
    help='Number of neurons in first layer',
    type=int
)

parser.add_argument(
    '--units2',
    default=12,
    help='Number of neurons in second layer',
    type=int
)

args = parser.parse_args()
