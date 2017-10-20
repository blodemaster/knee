import numpy as np
import tensorflow as tf
from scipy.ndimage import label as ndlabel
import scipy.io as sio
import os
from file_operation import find_corresponding_file_path


def channel_first2last(data):
    """Convert the given data from NCHW data format to NHWC data format"""
    return tf.transpose(data, [0, 3, 1, 2])


def channel_last2first(data):
    """Convert the given data form NHWC to NCHW data format"""
    return tf.transpose(data, [0, 2, 3, 1])


def per_image_standardization(image):
    """Standarize/normalize image to have zero mean and unit norm.

    The denominator is max(std, 1.0/sqrt(# of voxels of image)
    """
    mean = np.mean(image)

    std = np.std(image)
    a, b, c = image.shape
    num_of_oveall_voxels = a * b * c
    adjusted_std = max(std, 1.0 / np.sqrt(num_of_oveall_voxels))

    return (image - mean) / adjusted_std


def image_padded(image, patch_size):
    """Return image with pad

    Args:
        image: 3d numpy array
        patch_size: int
    """
    min_val = np.min(image)
    return np.pad(image, patch_size // 2, 'constant', constant_values=min_val)


def convert_coord_with_pad(idx, patch_size):
    """Return the map from original coordinates system to new coordinates system.

    Args:
        idx: numpy array
            Original coordinates system
        patch_size: int
            The size of patch which defines the size of pad been added.
    Return:
        new_array: numpy array
            Mapped coordinates for original voxels

    After adding pad to the original image, the coordinate of certain pixel(voxel) will change in the new image
    Given original index, return corresponding index in new image
    """
    edge = patch_size // 2
    new_array = idx + edge
    return new_array


def shuffle(data, label, **kwargs):
    """Shuffle the data and its corresponding labels together

    Arguments:
        data: numpy array
            data want to be shuffled
        label: numpy array
            correspondent label
    Return:
        data and label after being shuffled
    """
    assert(data.shape[0] == label.shape[0])

    n = data.shape[0]
    idx = np.random.permutation(n)

    data_shuffled = data[idx, :, :, :]
    label_shuffled = label[idx]
    if 'coordinate' in kwargs:
        coord_shuffled = kwargs['coordinate'][:, idx]
        return data_shuffled, label_shuffled, coord_shuffled
    return data_shuffled, label_shuffled


def get_largest_component(lab):
    """Get largest connected component.

    Given a multi-class labeling,
    leave the largest connected component
    for each class.
    """
    classes = np.unique(lab)
    classes = np.delete(classes, np.argwhere(classes == 0))
    pruned_lab = np.zeros(lab.shape, dtype=lab.dtype)
    for c in classes:
        print("Finding largest connected component in class {}".format(c))
        # make it black and white
        bw = np.zeros(lab.shape)
        bw[lab == c] = 1
        # 26 connectivity for 3D images
        conn = np.ones((3,3,3))
        # clustered_lab.shape = bw.shape
        clustered_lab, n_comps = ndlabel(bw, conn)
        # sort components by volume from smallest to largest (skip zero)
        comp_volumes = [np.sum(clustered_lab == i) for i in range(1, n_comps)]
        comp_labels = np.argsort(comp_volumes)
        # pick component with largest volume (not counting background)
        largest_comp_label = 1 + comp_labels[-1]

        # keep the component in the output
        pruned_lab[clustered_lab==largest_comp_label] = c
    return pruned_lab


def save_rebuild_label(coordinates, labels, shape, path, output_dir):
    """Rebuilt the image with given labels value and save it.

    Arguments:
        coordinates: numpy array((3, None))
            The coordinates of voxels
        labels: numpy array(3d array)
            Voxels' correspondent labels(generated from prediction)
        shape: tuple
            The shape of the image
        path: string
            The path of input data
        output_dir: string
            The directory for output data

    Build an image with extracting largest components given provided labels' value as given position and save it as .mat file.
    """
    img = np.zeros(shape)
    for i in range(coordinates.shape[1]):
        x, y, z = coordinates[:, i]
        img[x, y, z] = labels[i]

    largest_img = get_largest_component(img)
    data = {}
    data['label'] = largest_img

    full_path = find_corresponding_file_path(path, output_dir)
    if not os.path.exists(os.path.dirname(full_path)):
        os.makedirs(os.path.dirname(full_path))
    sio.savemat(full_path, data)


def dice_score(real_labels, predict_labels, nb_classes=2):
    """Return Dice Similarity Coefficient (DSC) for each class.

    Arguments:
        real_labels: numpy array(2d array(None, 3))
            It is in categorical form(i.e. [0, 0, 1] for label=2)
        predict_labels: numpy array
            predict labels got from model, share same features as real labels
        nb_classes: int
            number of class you want to identify
    Return:
        Dice score: list(len=nb_classes)
    """
    print("background voxels: {}".format(np.where(predict_labels == 0)[0].shape))
    print("region 1: {}".format(np.where(predict_labels == 1)[0].shape))
    print('region 2: {}'.format(np.where(predict_labels == 2)[0].shape))

    cm = np.zeros((nb_classes, 3), dtype=np.float32)
    for c in range(nb_classes):
        pred = (predict_labels == c)
        real = (real_labels == c)

        cm[c, 0] = np.sum(pred * real)
        cm[c, 1] = np.sum(pred)
        cm[c, 2] = np.sum(real)

    return 2 * cm[:, 0] / (cm[:, 1] + cm[:, 2])


def accuracy(predicted_lab, true_lab):
    """Return the accuracy between predicted labels and true labels."""
    # predicted_lab = np.argmax(predicted_lab, axis=1)
    # true_lab = np.argmax(true_lab, axis=1)
    comp = (predicted_lab == true_lab)
    return np.sum(comp) / float(len(true_lab.ravel()))


if __name__ == "__main__":
    import scipy.io as sio
    data = sio.loadmat('ScanManTrain/9003406_20041118_SAG_3D_DESS_LEFT_016610296205.mat')
    image = data['scan']
    new = per_image_standardization(image)
    print(np.mean(new))
    print(np.std(new))