# !/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
import numpy as np
# import matplotlib.pyplot as plt
from fizz_buzz_net import FizzBuzzNet
from pyfunt.solver import Solver as Solver

import argparse

np.random.seed(0)

EXPERIMENT_PATH = '../Experiments/' + str(uuid.uuid4())[-10:]

# solver constants
NUM_PROCESSES = 1

NUM_TRAIN = 50000
NUM_TEST = 100

WEIGHT_DEACY = .00
REGULARIZATION = 0
LEARNING_RATE = .05
MOMENTUM = .9
NUM_EPOCHS = 2000
BATCH_SIZE = 128
CHECKPOINT_EVERY = 1000

NUM_DIGITS = 12

args = argparse.Namespace()


def parse_args():
    """
    Parse the options for running the Residual Network on CIFAR-10.
    """
    desc = 'Train a Residual Network on CIFAR-10.'
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument

    add('--experiment_path',
        metavar='DIRECOTRY',
        default=EXPERIMENT_PATH,
        type=str,
        help='directory where results will be saved')
    add('-load', '--load_checkpoint',
        metavar='DIRECOTRY',
        default='',
        type=str,
        help='load checkpoint from load_checkpoint')
    add('--n_processes', '-np',
        metavar='INT',
        default=NUM_PROCESSES,
        type=int,
        help='Number of processes for each step')
    add('--n_train',
        metavar='INT',
        default=NUM_TRAIN,
        type=int,
        help='Number of total samples to select for training')
    add('--n_test',
        metavar='INT',
        default=NUM_TEST,
        type=int,
        help='Number of total samples to select for validation')
    add('-wd', '--weight_decay',
        metavar='FLOAT',
        default=WEIGHT_DEACY,
        type=float,
        help='Weight decay for sgd_th')
    add('-reg', '--network_regularization',
        metavar='FLOAT',
        default=REGULARIZATION,
        type=float,
        help='L2 regularization term for the network')
    add('-lr', '--learning_rate',
        metavar='FLOAT',
        default=LEARNING_RATE,
        type=float,
        help='Learning rate to use with sgd_th')
    add('-mom', '--momentum',
        metavar='FLOAT',
        default=MOMENTUM,
        type=float,
        help='Nesterov momentum use with sgd_th')
    add('--n_epochs', '-nep',
        metavar='INT',
        default=NUM_EPOCHS,
        type=int,
        help='Number of epochs for training')
    add('--batch_size', '-bs',
        metavar='INT',
        default=BATCH_SIZE,
        type=int,
        help='Number of images for each iteration')
    add('--checkpoint_every', '-cp',
        metavar='INT',
        default=CHECKPOINT_EVERY,
        type=int,
        help='Number of epochs between each checkpoint')
    parser.parse_args(namespace=args)
    assert not (args.network_regularization and args.weight_decay)


def binary_encode(i, num_digits):
    # TODO PuT MSB to the right

    return np.array([i >> d & 1 for d in range(num_digits)])


def binary_decode(bitlist):
    out = 0
    for bit in bitlist[::-1]:
        out = (out << 1) | bit
    return out


def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


def custom_update_decay(epoch):
    return 1


def main():
    parse_args()

    X = np.array([binary_encode(i, NUM_DIGITS) for i in range(2**NUM_DIGITS)])
    Y = np.array([fizz_buzz_encode(i) for i in range(2**NUM_DIGITS)])

    Xte = X[:101]
    Yte = Y[:101]

    # np.random.shuffle(mask)

    Xtr = X[101:2**NUM_DIGITS]
    Ytr = Y[101:2**NUM_DIGITS]

    Xval = X[:101]
    Yval = Y[:101]

    data = {
        'X_train': Xtr,
        'y_train': Ytr,
        'X_val': Xval,
        'y_val': Yval,
    }

    reg = args.network_regularization

    model = FizzBuzzNet(reg=reg, input_dim=NUM_DIGITS, hidden_dims=[100, 100])

    wd = args.weight_decay
    lr = args.learning_rate
    mom = args.momentum

    optim_config = {'learning_rate': lr, 'nesterov': False,
                    'momentum': mom, 'weight_decay': wd}

    epochs = args.n_epochs
    bs = args.batch_size
    num_p = args.n_processes
    cp = args.checkpoint_every

    solver = Solver(model, data, args.load_checkpoint,
                    num_epochs=epochs, batch_size=bs,  # 20
                    update_rule='sgd',
                    optim_config=optim_config,
                    custom_update_ld=custom_update_decay,
                    checkpoint_every=cp,
                    num_processes=num_p,
                    check_and_swap_every=200,
                    silent_train=True)

    solver.train()
    predictions = solver.check_accuracy(Xte, return_preds=True)
    print 'Accuracy: %f' % np.mean(predictions == Yte)
    Xints = np.array([binary_decode(i) for i in Xte])

    output = np.vectorize(fizz_buzz)(Xints, predictions)
    print list(output)

    # solver.export_model(exp_path)
    # solver.export_histories(exp_path)

    print 'finish'


if __name__ == '__main__':
    main()
