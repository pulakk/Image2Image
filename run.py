import os, argparse
import matplotlib.pyplot as plt 
from keras.models import load_model

from image_data import ImageGenerator, plot_multiple
from models import Im2Im


def run(path_x, path_y, path_y_, path_model, 
        testing, retrain, grayscale, size, epochs, batch_size):

    shape = (size, size, 1 if grayscale else 3)

    generator_x = ImageGenerator(path_x, batch_size, shape)
    generator_y = ImageGenerator(path_y, batch_size, shape)

    if testing:
        im2im = load_model(path_model)
    else:
        if retrain:
            im2im = Im2Im(shape)
        else:
            try:
                im2im = load_model(path_model)
            except Exception as e:
                print('\n',e,'\n')
                im2im = Im2Im(shape)

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
            plot_multiple(Y)
            plt.savefig(os.path.join(path_y_,str(image_name)+'_Y.png'))
            plt.close()
            plot_multiple(Y_)
            plt.savefig(os.path.join(path_y_,str(image_name)+'_Y_.png'))
            plt.close()
            
            print('Epoch:',epoch,'>>',batch_id*100//n_batches,'% : ','loss = %.3f' % (loss),end=end)


        if not testing:
            print('Epoch:',epoch,'>>',' loss = %.3f\t\t' % (loss))
            im2im.save(path_model)
        

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_file_path',help = 'Path to model (including file name and format)')
    parser.add_argument('input_folder', help = 'folder path for input images')
    parser.add_argument('output_folder', help = 'folder path for output images')
    parser.add_argument('predictions_folder', help = 'folder path for predicted images')

    parser.add_argument('-t','--testing', action = 'store_true', help = 'whether testing model')
    parser.add_argument('-r','--retrain', action = 'store_true', help = 'retrain model again')
    parser.add_argument('-g','--grayscale',action = 'store_true', help='whether to train only for gray images')

    parser.add_argument('-s','--size',type=int, default = 32, help='size (width/height) of image')

    parser.add_argument('-e','--epochs',type=int, default = 1000, help='number of epochs to train')
    parser.add_argument('-b','--batch_size',type=int, default = 16, help='training batch size')

    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        raise OSError('Folder not found :',args.input_folder)
    
    if not os.path.exists(args.output_folder):
        raise OSError('Folder not found :',args.output_folder)

    if not os.path.exists(args.predictions_folder):
        raise OSError('Folder not found :',args.predictions_folder)

    print(args)

    run(
        args.input_folder, 
        args.output_folder, 
        args.predictions_folder, 
        args.model_file_path,
        args.testing,
        args.retrain,
        args.grayscale,
        args.size,
        args.epochs,
        args.batch_size
    )