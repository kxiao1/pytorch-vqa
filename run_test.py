from datetime import datetime
import time


def input_or_default(message, default):
    res = input(message)
    if res is None or res == "":
        res = default
    return res

def request_parameters():
    res = {}

    res["weight_decay"] = input_or_default("Weight decay (DEFAULT: 0.01): ", 0.01)
    res["learning_rate"] = input_or_default("Learning rate (DEFAULT: 1e-3): ", 1e-3)
    res["augmented"] = input_or_default("Augmented (Y/N) (DEFAULT: N): ", False) == "Y"
    res["batch_size"] = input_or_default("Batch size (DEFAULT: 128): ", 128)
    res["model_to_use"] = input_or_default("Model (baseline/big/combined): ", "big").lower()
    res["max_answers"] = input_or_default("Max answers (DEFAULT: 3000): ", 3000)
    res["epochs"] = input_or_default("Epochs (DEFAULT: 30): ", 30)
    res["name"] = input("Name of test: ")

    return res

# Dictionary: {
#   'weight_decay': <...>,
#   'learning_rate': <...>,
#   ...,
#   'max_answers': <...>
# }
# weight_decay: Numerical (default = 0.01)
# learning_rate: Numerical (default = 1e-3)
# augmented: Y/N (default = No)
# batch_size: Numerical (default = 128)
# name: String (required)
# model_to_use: baseline/big
# max_answers: Numerical (default = 3000)
# epochs: Numerical (default = 30)
def generate_config_file(dictionary):
    qa_path = "vizwiz/Annotations_augmented" if dictionary["augmented"] else "vizwiz/Annotations_all"
    train_path = "vizwiz/train_augmented" if dictionary["augmented"] else "vizwiz/train_all"
    val_path = "vizwiz/val_augmented" if dictionary["augmented"] else "vizwiz/val_all"
    test_path = "vizwiz/test_all"
    preprocessed_path = "./resnet-14x14-aug.h5" if dictionary["augmented"] else "./resnet-14x14.h5"

    if int(dictionary["max_answers"]) == 3000:

        # vocabulary_path = "vocab-aug-3000.json" if dictionary["augmented"] else "vocab-3000.json"
        vocabulary_path = "vocab-3000.json"
    elif int(dictionary["max_answers"]) == 1000:
        vocabulary_path = "vocab-1000.json"
        # vocabulary_path = "vocab-aug-1000.json" if dictionary["augmented"] else "vocab-1000.json"
    else:
        print("wrong max answers, needs to be 1000 or 3000, please try again")
        quit()
    colors_path = "colors.txt"
    unprocessed_images_path = "./unprocessed.h5"

    task = 'OpenEnded'
    dataset = 'mscoco'

    # preprocess config
    preprocess_batch_size = 64
    image_size = 448  # scale shorter end of image to this size and centre crop
    output_size = image_size // 32  # size of the feature maps after processing through a network
    output_features = 2048  # number of feature maps thereof
    central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

    # training config
    epochs = dictionary["epochs"]
    batch_size = dictionary["batch_size"]
    initial_lr = dictionary["learning_rate"] # default Adam lr
    lr_halflife = 50000  # in iterations
    data_workers = 8
    max_answers = dictionary["max_answers"]

    weight_decay = dictionary["weight_decay"]
    model_to_use = dictionary["model_to_use"]


    res = f"""
qa_path = '{qa_path}'  # directory containing the question and annotation jsons
train_path = '{train_path}'  # directory of training images
val_path = '{val_path}'  # directory of validation images
test_path = '{test_path}'  # directory of test images
preprocessed_path = '{preprocessed_path}'  # path where preprocessed features are saved to and loaded from
vocabulary_path = '{vocabulary_path}'  # path where the used vocabularies for question and answers are saved to
colors_path = '{colors_path}' # path where list of colors is stored
unprocessed_images_path = '{unprocessed_images_path}'
test_preprocessed_path = 'resnet-test.h5'

task = '{task}'
dataset = '{dataset}'

# preprocess config
preprocess_batch_size = {preprocess_batch_size}
image_size = {image_size}  # scale shorter end of image to this size and centre crop
output_size = {image_size} // 32  # size of the feature maps after processing through a network
output_features = {output_features}  # number of feature maps thereof
central_fraction = {central_fraction}  # only take this much of the centre when scaling and centre cropping

# training config
epochs = {epochs}
batch_size = {batch_size}
initial_lr = {initial_lr}  # default Adam lr
lr_halflife = {lr_halflife}  # in iterations
data_workers = {data_workers}
max_answers = {max_answers}

# weight decay
weight_decay = {weight_decay}
model_to_use = '{model_to_use}'
    """

    res_dict = {
        "qa_path": "vizwiz/Annotations_augmented" if dictionary["augmented"] else "vizwiz/Annotations_all",
        "train_path": "vizwiz/train_augmented" if dictionary["augmented"] else "vizwiz/train_all",
        "val_path": "vizwiz/val_augmented" if dictionary["augmented"] else "vizwiz/val_all",
        "test_path": "vizwiz/test_augmented" if dictionary["augmented"] else "vizwiz/test_all",
        "preprocessed_path": "./resnet-14x14-aug.h5" if dictionary["augmented"] else "./resnet-14x14.h5",
        "vocabulary_path": "vocab-aug.json" if dictionary["augmented"] else "vocab.json",
        "colors_path": "colors.txt",
        "unprocessed_images_path": "./unprocessed.h5",

        "task": 'OpenEnded',
        "dataset": 'mscoco',

        # preprocess config
        "preprocess_batch_size": 64,
        "image_size": 448,  # scale shorter end of image to this size and centre crop,
        "output_size": image_size // 32,  # size of the feature maps after processing through a network
        "output_features": 2048,  # number of feature maps thereof,
        "central_fraction": 0.875,  # only take this much of the centre when scaling and centre cropping,

        # training config
        "epochs": dictionary["epochs"],
        "batch_size": dictionary["batch_size"],
        "initial_lr": dictionary["learning_rate"], # default Adam lr,
        "lr_halflife": 50000,  # in iterations,
        "data_workers": 8,
        "max_answers": dictionary["max_answers"],

        "weight_decay": dictionary["weight_decay"],

    }

    return res, res_dict

def main():
    parameters = request_parameters()
    print(parameters)

    config_file, config_dict = generate_config_file(parameters)
    print(config_file)

    with open("config.py", "w+") as f:
        f.write(config_file)

    time_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    name = parameters["name"]
    with open(f"logs_info/{name}_{time_now}_config.py", "w+") as f:
        f.write(config_file)

    import model_big
    import model_baseline
    import model_combined
    time.sleep(2)

    # actually run the model
    if parameters["model_to_use"] == "baseline":
        print("Running baseline model.")
        model_baseline.main(parameters["name"])
    elif parameters["model_to_use"] == "big":
        print("Running big model.")
        model_big.main(parameters["name"])
    else:
        print("Running combined model.")
        model_combined.main(parameters["name"])

if __name__ == "__main__":
    main()
