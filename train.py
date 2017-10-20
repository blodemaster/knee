import numpy as np
from network import TriplanarCNN
from sampling import extract_patches_from_image
from utility import shuffle
from keras.utils import np_utils
import time

def get_data(image_paths, label_name, patch_size, nb_samples):
    """Return data(containing patches and labels) for training the model

    Args:
        image_paths: list
            Paths for all training data
        label_name: string
            name of label of interest
        patch_size: int
        nb_samples: int
            The number of samples extracted for each image
    """
    data = []
    labels = []

    for path in image_paths:
        x, y = extract_patches_from_image(path, label_name, patch_size, nb_samples)
        data.append(x)
        labels.append(y)

    data = np.concatenate(data)
    labels = np.concatenate(labels)

    data, labels = shuffle(data, labels)

    return data, labels


def train_model(image_paths, label_name, patch_size, nb_classes,
                nb_samples, model_path, statistic_path, log_path, init,
                optimizer, network, batch_size=128, epoch=300):
    """Train the model with given parameters and return model

    Args:
        image_paths: list of string
            The list contains path for all training dataset.
        label_name: string
            The name of the structure of interest
        patch_size: int
            The size of patch
        nb_classes: int
            the number of class
        nb_samples: int
            The number of voxels that will be picked for training for each file
        model_path: string
            The path of model
        statistic_path: string
            The path of statistic(i.e, mean and standard deviation for training data)
        log_path: string
            The path of log which log training process
        init: string
            The name of algorithm used for weight initialization
        optimizer: keras.optimizer
            The optimizer used for backpropagation
        batch_size: int
            The size of samples been passed to model at a time
        epoch: int
            The number of iteration
    """
    if network == 'TriplanarCNN':
        net = TriplanarCNN(patch_size, nb_classes)

    net.set_callbacks(model_path=model_path, log_path=log_path, stats_path=statistic_path)
    net.build()

    t1 = time.time()

    images, labels = get_data(image_paths, label_name, patch_size, nb_samples)

    labels = np_utils.to_categorical(labels, nb_classes)

    print(images.shape)
    print("The preprocessing of training data takes {} seconds".format(time.time() - t1))
    print("\n")

    t2 = time.time()

    net.compile(optimizer=optimizer, init=init)
    net.train(images, labels,
              batch_size=batch_size, epoch=epoch)

    net.save_stats(statistic_path)

    print("Training model is done in {} seconds".format(time.time() - t2))
    print("\n")

if __name__ == "__main__":
    img_path = ['train_data/9003406_20041118_SAG_3D_DESS_LEFT_016610296205.mat']
    label_dir = 'MedialTibialCartilage'
    patch_size = 28
    nb_classes = 2
    nb_samples = 2000
    model_path = './model.h5'
    statistic_path = './statistic.h5'
    log_path = './log.csv'
    train_model(img_path, label_dir, patch_size, nb_classes, model_path, statistic_path, log_path,)


