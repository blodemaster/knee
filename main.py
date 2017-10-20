import glob, os
from keras.optimizers import Adam
from train import train_model
from test_data_generator import test_data_generator
from evaluation import evaluate_performance


model_path = './output/v1_samples_30000/model.h5'
statistic_path = './output/v1_samples_30000/statistic.h5'
log_path = './output/v1_samples_30000/log.csv'
if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))

init = 'he_uniform'
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
patch_size = 28
nb_classes = 2

nb_training_samples = 30000

training_data_paths = glob.glob('train_data/*.mat')
# training_data_paths = ['./train_data/9003406_20041118_SAG_3D_DESS_LEFT_016610296205.mat']
test_data_paths = glob.glob('valid_data/*.mat')
# test_data_paths = ['./valid_data/9007827_20041006_SAG_3D_DESS_LEFT_016610263603.mat']

output_dir = './output/v1_samples_30000/'


network = 'TriplanarCNN'
label_name = 'MedialTibialCartilage'
retrain = False
mask_dir = 'valid_mask'

gpu_batch_size = 256

# train the model
if retrain:
    train_model(training_data_paths, label_name, patch_size,
                nb_classes, nb_training_samples, model_path, statistic_path,
                log_path, init, optimizer, network, batch_size=256)

# make prediction on test data and rebuild the image
test_data_generator(test_data_paths, output_dir, mask_dir,
                    patch_size, nb_classes, gpu_batch_size,
                    model_path, network)

# evaluate the performance of model
predict_pathes = glob.glob(output_dir + '*.mat')
#
output_result = './output/v1_samples_30000/performance.txt'
test_label_dir = "valid_data/"
evaluate_performance(predict_pathes, label_name, test_label_dir, output_result)
