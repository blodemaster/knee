from sampling import create_boundary
from file_operation import find_corresponding_file_path
import scipy.io as sio
import glob, os


label_name = 'MedialTibialCartilage'
target_paths = glob.glob('valid_data/*.mat')

output_dir = 'valid_mask/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for path in target_paths:
    data = sio.loadmat(path)
    labels = data[label_name].squeeze()
    regions = [1]

    mask_data = create_boundary(labels, regions)

    mask_data[mask_data > 0] = 1
    mask = {}
    mask['label'] = mask_data

    output_path = find_corresponding_file_path(path, output_dir)

    sio.savemat(output_path, mask)
