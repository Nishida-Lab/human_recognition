#python library
import argparse
import time
import numpy as np
import getpass
import shutil
import os
import gc

# PIL
from PIL import Image, ImageCms

# OpenCV
import cv2

#chainer libraty
import chainer
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import serializers

from chainer import cuda, Function, gradient_check, Variable, optimizers, utils
from chainer.functions.loss.mean_squared_error import mean_squared_error
from chainer.functions.loss.sigmoid_cross_entropy import sigmoid_cross_entropy

from chainer.dataset.convert import concat_examples


#python scripts
import load_dataset as ld
import network_structure as nn


# load image data from file name
def myConverter(batch, device, padding=None):

    #size = 100
    size = 80
    #size = 32

    newBatch = []
    del newBatch
    newBatch = []

    for i in range(len(batch)):

        #print batch[i]

        x = cv2.cvtColor(cv2.imread(batch[i][0]),cv2.COLOR_BGR2RGB)
        y = batch[i][1]

        x = cv2.resize(x, (size,size), interpolation = cv2.INTER_LINEAR)

        x = x.reshape((3,size,size)).astype(np.float32) / 255.
        #x = np.rollaxis(x.reshape((3,size,size)),0,3)
        #x = x.reshape((3,size,size)).astype(np.float32) / 255.

        newBatch.append((x, y))

        del x, y
        gc.collect()

    return concat_examples(newBatch, device=device, padding=None)


#Main
if __name__ == '__main__':

    # Load CNN model
    # model = nn.CNN_thibault2()
    model = nn.CNN_thibault_ultimate()

    # Setup optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_N = 500000
    validation_N = 20000

    # parse args
    parser = argparse.ArgumentParser(description='CIFAR10 CLASSIFER')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=5,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load dataset
    print "\n"+"loading the dataset.."+"\n"

    Xt, Yt, Xv, Yv = ld.load_dataset(train_N, validation_N)

    train = tuple_dataset.TupleDataset(Xt, Yt)
    validation = tuple_dataset.TupleDataset(Xv, Yv)

    print "\n"+"the dataset is loaded!!"+"\n"
    print "train_N: " + str(len(Xt))
    print "validation_N: " + str(len(Xv)) + "\n"
    print 'minibatch-size: ' + format(args.batchsize)
    print 'epoch: ' + format(args.epoch) + "\n"

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        print 'using GPU \n'

    start_time = time.time() #start time measurement

    # train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    # validation_iter = chainer.iterators.SerialIterator(validation, args.batchsize, repeat=False)

    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize, n_processes=args.loaderjob)
    validation_iter = chainer.iterators.MultiprocessIterator(validation, args.batchsize, repeat=False, n_processes=args.loaderjob)

    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu, converter=myConverter)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.Evaluator(validation_iter, model, device=args.gpu, converter=myConverter))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    #trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss','main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()

    execution_time = time.time() - start_time

    print "execution time : " + str(execution_time)

    if args.gpu >= 0:
        model.to_cpu()

    serializers.save_npz('cnn_gpu.model', model)
    #serializers.save_npz('cnn_gpu.state', optimizer)
    print "model saved"
