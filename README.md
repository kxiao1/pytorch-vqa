# How to run and use the code
This project was completed together with [Chittesh Thavamani](https://github.com/tchittesh) and [Andrew Rausch](https://github.com/rauschaj) for an introductory ML class. Following is an explanation of the various components of our code, along with a tutorial on how to use the code to run various models. Unless otherwise specified, we developed each of the files noted. Please refer to `README_original.md` for a specification of the original, 'Baseline' model.

## Getting started

In order to get started with training and testing our models, a few preliminary steps must be taken. These include:

2. Download and extract the training, validation, and test data and annotations from the VizWiz website. These should be placed in the `vizwiz/` directory and named corresponding to the appropriate fields in `config.py` (which is modified slightly from the forked repository).
3. Preprocess images and vocabulary by running `preprocess-images.py` and `preprocess-vocab.py`. Both of these files are from the original forked repository. This is necessary to generate the resnet features and vocabulary file before training any models. If running the Naive model, it is also necessary to run `preprocess-images-dumb.py`.
4. If using augmented data, run `transform.py`. This is a utility which takes the images in the `vizwiz/` directory, augments them, and saves further images back to this directory. Within the file you can specify parameters detailing the range of transformations that should be taken along with the number of augmented images to generate per training image.

With these steps finished, you are ready to run the model!

## Models

The following models are present in the code:
- Naive model (`model_degenerate.py`): Naive linear model as described in the research paper, which consists of a simple combination of LSTM, CNN, and fully-connected layer.
- Baseline model (`model_baseline.py`): Reference implementation of the baseline model, provided from the forked GitHub repsitory.
- Individual features
    - _Is color_ feature (`model_is_color.py`): Architecture to determine, provided a question, whether it pertains to color.
    - _Suitability_ feature (`model_suitable.py`): Architecture to determine, provided an image, whether it is suitable.
    - _Color_ feature (`model_colors.py`): Architecture to determine, provided an image, what the most relevant color in that image is.
- Big model (`model_big.py`): Baseline model combined with all individual features at the Concat layer, as explained in the research paper.
- Modified attention model (`model_modified_attention.py`): Baseline model augmented with a modified attention layer as described in the research paper.
- Combined model (`model_combined.py`): Combination of the big and modified attention models in the sense that the modified attention layer is used, _and_ each of the individual features are combined at the Concat layer as in the big model.

In our paper, each of these models were run with a weight decay of 1e-5, batch size of 128, and learning rate of 1e-3. Models run without an augmented dataset were run for 30 epochs; models run with an augmented dataset were run for 12 (because the augmented dataset contained more training examples per epoch).

## Training and evaluation

The easiest way of training the model is to use `run_test.py`, which provides a wizard to train a model with specified learning rate, weight decay, batch size, and so on. This generates a log file in the appropriate directory with the name provided when running the wizard.

Models can be evaluated in two ways: by their validation accuracy performance and their test accuracy performance. Validation accuracy may be tested locally, as epoch-over-epoch training and validation loss and accuracy are stored inside the saved log (`.pth`) file. To plot validation accuracy, you can use the `scripts/visualize_tracker.py` utlity and pass in the log file of the model you seek to evaluate as a command-line argument. This utility provides basic statistics about the validation accuracy of your model (for example, the average validation accuracy over the final ten epochs), along with a graph generated by `matplotlib`, which is saved to the `img/` directory.

The other way to evaluate a model is to run it on the test dataset, for which accuracy can only be determined by submission to the VizWiz server. To determine test accuracy, you should run `test.py <type> <log file>` where `type` is the type of model corresponding to the log file (for example, `baseline`, `naive`, or `big`). This will generate a `.json` file in the `test_results/` directory of the predictions of the model on the test dataset. This file can then be uploaded to the VizWiz competition portal at `eval.ai` to determine test accuracy.