import os, argparse
import matplotlib.pyplot as plt 
from keras.models import load_model
import numpy as np

from image_data import ImageGenerator, plot_multiple, idct_2D
from models import Im2Im

def get_idct_samples(Y_, shape):
    Y__ = np.zeros(shape)
    for i in range(Y_.shape[0]):
        for j in range(Y_.shape[1]):
            for k in range(Y_.shape[2]):
                Y__[i][j][k] = Y_[i][j][k]
        Y__[i] = idct_2D(Y__[i])
    
    return Y__

def run(path_x, path_y, path_y_, path_model, 
        testing, replace, grayscale, size, epochs, batch_size, dct_size):

    if testing:
        im2im = load_model(path_model)
        im2im.summary()
        print("Saved model loaded:", path_model)

        size = im2im.layers[0].input_shape[1]
        dct_size = im2im.layers[-1].output_shape[1]
        input_channels = im2im.layers[0].input_shape[3]
        output_channels = im2im.layers[-1].output_shape[3]

        in_shape = (size, size, input_channels)
        out_shape = (dct_size, dct_size, output_channels)
    else:
        in_shape = (size, size, 1 if grayscale else 3)
        out_shape = (dct_size, dct_size, 1 if grayscale else 3)

        if replace:
            im2im = Im2Im(in_shape, out_shape)
            print("New Model created.")
        else:
            try:
                im2im = load_model(path_model)
                im2im.summary()
                print("Saved model loaded:", path_model)
            except Exception as e:
                print('\n',e,'\n')
                im2im = Im2Im(in_shape, out_shape)
            print("New Model created.")

    generator_x = ImageGenerator(path_x, batch_size, in_shape)
    generator_y = ImageGenerator(path_y, batch_size, in_shape, dct_size)


    for epoch in range(1 if testing else epochs):
        n_batches = len(os.listdir(path_x))//batch_size
        loss = 0

        for batch_id in range(n_batches):
            X = next(generator_x)
            Y = next(generator_y)
            Y_ = im2im.predict(X)

            loss = ((Y_ - Y)**2).mean(axis=None)

            if testing:
                image_name = batch_id
                end = '\n'
            else:
                im2im.train_on_batch(X, Y)
                image_name = epoch 
                end = '\r'

            plot_multiple(X)
            plt.savefig(os.path.join(path_y_,str(image_name)+'_X.png'))
            plt.close()
            # plot_multiple(Y)
            # plt.savefig(os.path.join(path_y_,str(image_name)+'_dct_Y.png'))
            # plt.close()
            # plot_multiple(Y_)
            # plt.savefig(os.path.join(path_y_,str(image_name)+'_dct_Y_.png'))
            # plt.close()

            plot_multiple(get_idct_samples(Y, X.shape))
            plt.savefig(os.path.join(path_y_,str(image_name)+'_Y.png'))
            plt.close()
            plot_multiple(get_idct_samples(Y_, X.shape))
            plt.savefig(os.path.join(path_y_,str(image_name)+'_Y*.png'))
            plt.close()

            print('Epoch:',epoch,'>>',batch_id*100//n_batches,'% : ','loss = %.3f     ' % (loss),end=end)

        if testing:
            print('Predictions saved to', path_y_)
        else:
            print('Epoch:',epoch,'>> 100%',' loss = %.3f\t\t' % (loss))
            im2im.save(path_model)
            print('Model saved to', path_model)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_file_path',help = 'Path to model (including file name and format)')
    parser.add_argument('input_folder', help = 'folder path for input images')
    parser.add_argument('output_folder', help = 'folder path for output images')
    parser.add_argument('predictions_folder', help = 'folder path for predicted images')

    parser.add_argument('-t','--testing', action = 'store_true', help = 'whether testing model')
    parser.add_argument('-r','--replace', action = 'store_true', help = 'Replace exising model.')

    parser.add_argument('-g','--grayscale', action = 'store_true', help='whether to train only for gray images')
    parser.add_argument('-s','--size', type=int, default = 32, help='size (width/height) of image')
    parser.add_argument('-d','--dct_size', type=int, default = 10, help='Select first `dct_size` portion of image `size` for output')

    parser.add_argument('-e','--epochs', type=int, default = 1000, help='number of epochs to train')
    parser.add_argument('-b','--batch_size', type=int, default = 16, help='training batch size')

    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        raise OSError('Folder not found :',args.input_folder)
    
    if not os.path.exists(args.output_folder):
        raise OSError('Folder not found :',args.output_folder)

    if not os.path.exists(args.predictions_folder):
        raise OSError('Folder not found :',args.predictions_folder)

    if args.dct_size > args.size:
        raise AssertionError('--dct_size (-d) cannot be greater than --size (-s)')

    print(args)

    run(
        args.input_folder, 
        args.output_folder, 
        args.predictions_folder, 
        args.model_file_path,
        args.testing,
        args.replace,
        args.grayscale,
        args.size,
        args.epochs,
        args.batch_size,
        args.dct_size
    )