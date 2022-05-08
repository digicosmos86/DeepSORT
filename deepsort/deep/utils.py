### This is taken from HW6 CNN code

import re
import os

import tensorflow as tf
import argparse



class CustomModelSaver(tf.keras.callbacks.Callback):
    """Custom Keras callback for saving weights of networks."""

    def __init__(self, checkpoint_dir, max_num_weights=5):
        super(CustomModelSaver, self).__init__()

        self.checkpoint_dir = checkpoint_dir
        self.max_num_weights = max_num_weights

    def on_epoch_end(self, epoch, logs=None):
        """At epoch end, weights are saved to checkpoint directory."""

        min_acc_file, max_acc_file, max_acc, num_weights = self.scan_weight_files()

        cur_acc = logs["val_sparse_categorical_accuracy"]

        # Only save weights if test accuracy exceeds the previous best
        # weight file
        if cur_acc > max_acc:
            save_name = "weights.e{0:03d}-acc{1:.4f}.h5".format(epoch, cur_acc)

            self.model.save_weights(self.checkpoint_dir / save_name)

            # Ensure max_num_weights is not exceeded by removing
            # minimum weight
            if self.max_num_weights > 0 and num_weights + 1 > self.max_num_weights:
                os.remove(self.checkpoint_dir / min_acc_file)

    def scan_weight_files(self):
        """Scans checkpoint directory to find current minimum and maximum
        accuracy weights files as well as the number of weights."""

        min_acc = float("inf")
        max_acc = 0
        min_acc_file = ""
        max_acc_file = ""
        num_weights = 0

        for weight_file in self.checkpoint_dir.glob("*.h5"):
            num_weights += 1
            file_acc = float(
                re.findall(r"[+-]?\d+\.\d+", weight_file.name.split("acc")[-1])[0]
            )
            if file_acc > max_acc:
                max_acc = file_acc
                max_acc_file = weight_file.name
            if file_acc < min_acc:
                min_acc = file_acc
                min_acc_file = weight_file.name

        return min_acc_file, max_acc_file, max_acc, num_weights


def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights.''')


    return parser.parse_args()