import scipy.io as sio
import numpy as np
from skimage.morphology import cube, dilation
from utility import (
    per_image_standardization,
    convert_coord_with_pad,
    image_padded
)


def create_boundary(labels, regions, width=10):
    """Return new label numpy array including labels for boundary zone

    Args:
        labels: 3d numpy array
            labels information
        regions: list
            list containing the label of region(tissue) that we are interested to do dilation on
            In my case, labels could be either 0, 1, and 0 represents background, so regions=[1]
        width: int
            how much distance would you like to dilate

    Find the boundary region for each region by using dilation method.
    Take the union of all separate boundary regions as target region and assign it with new label with label of given
    region + len(regions).
    """
    labels_dilated = labels.copy()
    nb_region = len(regions)

    id_protected = np.in1d(labels.ravel(), regions).reshape(labels.shape)

    kernel = cube(2 * width + 1)

    for region in regions:
        labels_binary = np.zeros(labels.shape, dtype=labels.dtype)
        labels_binary[np.where(labels == region)] = 1

        lab_boundary = dilation(labels_binary, kernel) - labels_binary
        idx_boundary = (lab_boundary == 1)

        labels_dilated[idx_boundary & ~id_protected] = region + nb_region

    return labels_dilated


def all_boundaries(labels_dilated, region_exclusive):
    """Return all the coordinates of voxels that are located within boundary zone

    Args:
        labels_dilated: 3d numpy array
            Matrix containing information of boundary region
        region_exclusive: list
            list contain the label that belongs to boundary region
    Return:
        result: 3d numpy array
    """
    all_coord = []
    for region in region_exclusive:
        idx = np.array(np.where(labels_dilated == region))
        all_coord.append(idx)
    if len(region_exclusive) == 1:
        return all_coord[0]
    else:
        result = np.hstack(all_coord)
        return result


def random_pick(idx, number_samples):
    """Return the coordinates of the voxels that are picked

    Args:
        idx: numpy array (3, None)
            the coordinates of the voxel for a label of interest
        number_samples: int
            the number of samples should be picked form given coordinates
    Return:
        numpy array(3, None)
        The coordinates info of voxels being chosen

    Randomly choose nb_choice size voxel from voxels.
    If the size of sampling is larger than the voxels we have, then first pick every voxel once and
    then choose the residues randomly from voxels with repetition allowable
    """
    length = idx.shape[1]
    if length > number_samples:
        result = np.random.choice(range(length), size=number_samples, replace=False)
    else:
        result = np.hstack((range(length), np.random.choice(range(length), number_samples-length, replace=True)))
    return idx[:, result]


def sampler(label, nb_samples):
    """Return coordinates of voxels that are extracted.

    Args:
        label: 3d numpy array
            Array contains original labels information
        nb_samples: int
            The number of voxel we want to get(should be the integral multiple of 3)
    Return:
        result_coordinates: numpy array(2d array:(3, None))
            a series of coordinates corresponding to the voxels that are extracted.

    pick given number of samples from voxels within target region.
    If the voxels of interest is background, first sample points evenly from the boundary of each tissue(region)
    then randomly pick the rest of required background voxels.
    """
    result = []
    labels_original = np.unique(label)

    per_samples, reminder = divmod(nb_samples, labels_original.shape[0])

    temp_region = labels_original.tolist()
    temp_region.remove(0)

    labels_dilated = create_boundary(label, temp_region)

    regions_exclusive = np.setdiff1d(labels_dilated, labels_original)
    # regions_exclusive = np.delete(regions_exclusive, 0)

    for flag in temp_region:
        idx = np.array(np.where(label == flag))
        result.append(random_pick(idx, per_samples))

    all_boundary_voxels = all_boundaries(labels_dilated, regions_exclusive)
    num_samples_background = per_samples + reminder

    result.append(random_pick(all_boundary_voxels, num_samples_background))
    result = np.concatenate(result, axis=1)
    return result


def extract_orthogonal_patches(image_padded, idx, patch_size):
    """Return triplanar patches of a given voxel

    Args:
        image_padded: 3d numpy array
            The image being padded
        idx: 1d numpy array
            the original coordinate of voxel of interest
        patch_size: int
            The size of patch
    Return:
        triplanars: 3d numpy array
            if patch_size = 28, the output size(3, 28, 28)
    """
    edge = patch_size // 2
    id = convert_coord_with_pad(idx, patch_size)
    # temp = []
    plane_1 = image_padded[id[0], id[1]-edge:id[1]+edge, id[2]-edge:id[2]+edge]
    plane_2 = image_padded[id[0]-edge:id[0]+edge, id[1], id[2]-edge:id[2]+edge]
    plane_3 = image_padded[id[0]-edge:id[0]+edge, id[1]-edge:id[1]+edge, id[2]]
    temp = [plane_1, plane_2, plane_3]
    # voxel_box = np.dstack((plane_1, plane_2, plane_3))
    voxel_box = np.concatenate([data[np.newaxis] for data in temp])

    return voxel_box


def extract_patches_from_image(data_path, label_name, patch_size, nb_samples):
    """Return data containing voxels intensity and its corresponding label.

    Args:
        data_path: string
            The path of the selected data which contains all of masks and scan
        label_name: string
            The name of label of interest
        patch_size: int
            The size of extracted patch
        nb_samples: int
            The overall number of patches you select from one image
    Returns:
        data_picked: 4d numpy array(NCHW or NHWC)
            information of all tri-planar patches
        label_picked: 1d numpy array
            label information for corresponding voxel
    """
    data = sio.loadmat(data_path)
    image = data['scan'].squeeze()

    label = data[label_name].squeeze()
    assert image.shape == label.shape

    image = per_image_standardization(image)
    image = image_padded(image, patch_size)

    coordinates = sampler(label, nb_samples)
    assert coordinates.shape[1] == nb_samples
    print('Number of voxels selected: ', coordinates.shape)

    patches_picked = []
    label_picked = label[coordinates[0, :], coordinates[1, :], coordinates[2, :]]

    for i in range(nb_samples):
        patches_picked.append(extract_orthogonal_patches(image, coordinates[:, i], patch_size))

    output_patches = np.array(patches_picked)

    return output_patches, label_picked


# def test_info(image_path):
#     """Return the coordinate of the non background voxel"""
#     data = sio.loadmat(image_path)['scam'].squeeze()
#     idx = np.array(np.where(data > 0))  # 2D array (3, None)
#     return idx


if __name__ == '__main__':

    # data = sio.loadmat('ScanManTrain/9003406_20041118_SAG_3D_DESS_LEFT_016610296205.mat')
    # label = data['MedialTibialCartilage'].squeeze()
    data_path = 'ScanManTrain/9003406_20041118_SAG_3D_DESS_LEFT_016610296205.mat'
    label_name = 'MedialTibialCartilage'
    patch_size = 28
    nb_samples = 10000
    patches, labels = extract_patches_from_image(data_path, label_name, patch_size, nb_samples)
    print(patches.dtype)
    print(labels.dtype)
    import tensorflow as tf
    sess = tf.Session()
    dataset = tf.contrib.data.Dataset.from_tensor_slices((patches, labels))
    dataset = dataset.batch(128)

    for _ in range(2):
        iterator = dataset.make_one_shot_iterator()
        # image_op, label_op = iterator.get_next()
        next_element = iterator.get_next()
        for _ in range(100):
            try:
                image, label = sess.run(next_element)
                print(image.shape)
            except tf.errors.OutOfRangeError:
                break
