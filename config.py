
qa_path = 'vizwiz/Annotations_augmented'  # directory containing the question and annotation jsons
train_path = 'vizwiz/train_augmented'  # directory of training images
val_path = 'vizwiz/val_augmented'  # directory of validation images
test_path = 'vizwiz/test_augmented'  # directory of test images
preprocessed_path = './resnet-14x14-aug.h5'  # path where preprocessed features are saved to and loaded from
vocabulary_path = 'vocab-3000.json'  # path where the used vocabularies for question and answers are saved to
colors_path = 'colors.txt' # path where list of colors is stored
unprocessed_images_path = './unprocessed.h5'
test_preprocessed_path = 'resnet-test.h5'
test_unprocessed_path = 'unprocessed-test.h5'
# test_preprocessed_path = 'unprocessed-test.h5'

task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 64
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = 448 // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
epochs = 12
batch_size = 128
initial_lr = 0.001  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8
max_answers = 3000

# weight decay
weight_decay = 0 
model_to_use = 'baseline'
    
