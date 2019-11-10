# Image to Image Neural Network - Sequential Model


|Input Image (32 x 32 x 3) | Output Image (32 x 32 x 3) | Predicted Output (32 x 32 x 3) |
|----------- | ------------ | --------------- |
|<img src="https://github.com/mizimo/Image2Image/raw/master/performance/pokemon/1_X.png">| <img src="https://github.com/mizimo/Image2Image/raw/master/performance/pokemon/1_Y.png"> | <img src="https://github.com/mizimo/Image2Image/raw/master/performance/pokemon/1_Y*.png"> |

|Input Image (32 x 32 x 1) | Output Image (32 x 32 x 1) | Predicted Output (32 x 32 x 1) |
|----------- | ------------ | --------------- |
|<img src="https://github.com/mizimo/Image2Image/raw/master/performance/depth/0_X.png">| <img src="https://github.com/mizimo/Image2Image/raw/master/performance/depth/0_Y.png"> | <img src="https://github.com/mizimo/Image2Image/raw/master/performance/depth/0_Y_.png"> |

# Training New Models

The main file to be used is `run.py`. Using the help argument `-h` we can view the various arguments which are available for training a model.
```
usage: run.py [-h] [-t] [-r] [-g] [-s SIZE] [-d DCT_SIZE] [-e EPOCHS]
              [-b BATCH_SIZE]
              model_file_path input_folder output_folder predictions_folder

positional arguments:
  model_file_path       Path to model (including file name and format)
  input_folder          folder path for input images
  output_folder         folder path for output images
  predictions_folder    folder path for predicted images

optional arguments:
  -h, --help            show this help message and exit
  -t, --testing         whether testing model
  -r, --retrain         retrain model again
  -g, --grayscale       whether to train only for gray images
  -s SIZE, --size SIZE  size (width/height) of image
  -d DCT_SIZE, --dct_size DCT_SIZE
                        Select first `dct_size` portion of image `size` for
                        output
  -e EPOCHS, --epochs EPOCHS
                        number of epochs to train
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        training batch size
```

New models can be trained in the following way
- Collect input images and output images in two separate folders - `input_folder` and `output_folder`. The corresponding input and output images should have the same names. 
- Create another folder for the predicted images - `predictions_folder`
- Use the command `python3 run.py path/to/model.h5 input/folder output/folder prediction/folder -s <image_size> -d <dct_size>` in the root folder. Use `-g` argument to train on grayscale images. 
- Run `python3 run.py path/to/model.h5 test/input/folder test/output/folder test/predictions/folder -t` for testing the performance on untrained test images

** Used python 3.7.4
