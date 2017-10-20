import time
import scipy.io as sio
import numpy as np
from network import TriplanarCNN
from sampling import extract_orthogonal_patches
from utility import image_padded, per_image_standardization, save_rebuild_label
from file_operation import find_corresponding_file_path


def test_data_generator(paths, out_dir, mask_dir,
                        patch_size, nb_classes, batch_size,
                        model_path, network):
    """Make prediction on test data and save the result

    Args:

    """
    if network == 'TriplanarCNN':
        net = TriplanarCNN(patch_size, nb_classes)
    
    net.load_model(model_path)
    print(net.model.summary())
    net.compile()
    
    for path in paths:
        start_time = time.time()
        print('Initializing {}'.format(path))
        
        image = sio.loadmat(path)['scan'].squeeze()
        
        mask_path = find_corresponding_file_path(path, mask_dir)
        mask = sio.loadmat(mask_path)['label'].squeeze()
        coordinates= np.array(np.where(mask > 0))
        
        shape = image.shape
        print(coordinates.shape)

        image = per_image_standardization(image)
        img_padded = image_padded(image, patch_size)

        test_data = []
        counter = 0
        prediction = []
        loop = 0

        for i in range(coordinates.shape[1]):
            if counter == batch_size:
                loop += 1
                print("This is the {} loop".format(loop))
                test_data = np.array(test_data)

                print("Current data shape is {}".format(test_data.shape))
                temp = net.predict_on_batch(test_data)
                temp = np.argmax(temp, axis=1)
                print("prediction shape is: ", temp.shape)
                prediction.append(temp)
                test_data = []
                counter = 0

            test_data.append(extract_orthogonal_patches(img_padded, coordinates[:, i], patch_size))
            counter += 1

        if counter != batch_size:
            for i in range(batch_size - counter):
                test_data.append(extract_orthogonal_patches(img_padded, coordinates[:, i], patch_size))

        test_data = np.array(test_data)

        temp = net.predict_on_batch(test_data)
        temp = np.argmax(temp, axis=1)
        prediction.append(temp)

        prediction = np.hstack(tuple(prediction))
        print('The shape of final prediction is: ', prediction.shape)
        # extra prediction at the end of the predict list will be ignored by save_rebuild_image
        save_rebuild_label(coordinates, prediction, shape, path,
                           out_dir)  # get largest components is done within this f
        print("{} is predicted and saved in {}s".format(path, time.time() - start_time))


if __name__ == '__main__':
    import glob

    pathes = glob.glob('dataset/vali_mri/*.nii')
    mask_dir = 'dataset/vali_mask/'
    out_dir = 'vali_rebuilt/'

    model_path = './vali_rebuilt/model.h5'
    statistic_path = './vali_rebuilt/statistic.h5'

    patch_size = 29
    nb_classes = 3

    batch_size = 100000

    test_data_generator(pathes, mask_dir, out_dir, patch_size, nb_classes, batch_size, model_path, statistic_path)

