import os, time
import numpy as np
import scipy.io as sio
from utility import accuracy, dice_score
from file_operation import find_corresponding_file_path


def evaluate_performance(predict_lab_pathes, label_name, real_lab_dir, result_path):
    """Return accuracy and dice score between predicted label and true label

    Arguments:
        predict_lab_pathes: list of string
            The list contains all path of predicted label data
        label_name: string
            The name of the mask in original file
        real_lab_dir: string
            The path of directory storing correct label information
        result_path: string
            The path for saving performance result
    """
    accuracy_list = []
    dice_score_list = []
    if os.path.exists(result_path):
        os.remove(result_path)

    f = open(result_path, "w")
    for path in predict_lab_pathes:
        real_label_path = find_corresponding_file_path(path, real_lab_dir)

        predict_label = sio.loadmat(path)['label'].squeeze()
        real_label = sio.loadmat(real_label_path)[label_name].squeeze()

        score = accuracy(predict_label, real_label)
        dice = dice_score(real_label, predict_label)
        print("The accuracy for {} is: {}\n".format(path, score))
        print("The dice score on {} is: {}\n".format(path, dice))

        f.write("The accuracy for {} is: {}\n".format(path, score))
        f.write("The dice score on {} is: {}\n".format(path, dice))

        accuracy_list.append(score)
        dice_score_list.append(dice)

    final_score = sum(accuracy_list) / len(accuracy_list)
    final_dice_score_array = np.array(dice_score_list)
    final_dice_score = np.mean(final_dice_score_array, axis=0)
    dice_score_std = np.std(final_dice_score_array, axis=0)

    f.write("mean accuracy is: {}\n".format(final_score))
    f.write('mean dice score is: {}\n'.format(final_dice_score))
    f.write('std dice score is: {}\n'.format(dice_score_std))
    f.close()

    print("Job done")