import os
import h5py


def find_corresponding_file_path(source_path, target_dir):
    """Find the path of the corresponding label/data for the given path of a file.

    Args:
        source_path: string
            the full path of the given file
        target_dir: string
            the target directory of the file
    Return:
        The path of new corresponding file.
    """
    name = os.path.split(source_path)[-1]
    target_path = os.path.join(target_dir, name)
    return target_path


def write_h5file(outpath, data_dict, attr_dict=None):
    """ Write a dictionary into HDF5 file.

    Arguments:
        outpath: string
            The output files path
        data_dict: dictionary
            The dictionary to be written
        attr_dict: dictionary
            A dictionary containing general attributes
    """
    attr_dict = attr_dict if attr_dict else {}
    f = h5py.File(outpath, "w")
    for key in data_dict:
        f.create_dataset(name=key, data=data_dict[key])

    for key in attr_dict:
        f.attrs[key] = attr_dict[key]
    f.close()

def read_h5file(inpath):
    """ Read data from HDF5 file and return dictionary.

    Arguments:
        inpath: string
            The path of input HDF5 file

    Return:
        a dictionary containing the data
    """
    f = h5py.File(inpath, 'r')
    data = {}
    for key in f.keys():
        data[key] = f[key].value
    for key in f.attrs:
        data[key] = f.attrs[key]
    return data

