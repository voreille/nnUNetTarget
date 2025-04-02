import matplotlib
from batchgenerators.utilities.file_and_folder_operations import join

matplotlib.use("agg")
import seaborn as sns
import matplotlib.pyplot as plt

from nnunetv2.training.logging.nnunet_logger import nnUNetLogger


class nnUNetLoggerV1(nnUNetLogger):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """

    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            "mean_fg_dice": list(),
            "ema_fg_dice": list(),
            "dice_per_class_or_region": list(),
            "train_losses": list(),
            "val_losses": list(),
            "encoder_lrs": list(),
            "decoder_lrs": list(),
            "epoch_start_timestamps": list(),
            "epoch_end_timestamps": list(),
        }
        self.verbose = verbose
        # shut up, this logging is great

    def plot_progress_png(self, output_folder):
        # Infer the epoch from internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1
        sns.set(font_scale=2.5)
        fig, ax_all = plt.subplots(
            4, 1, figsize=(30, 72)
        )  # Added one more subplot for additional LR

        # Regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(
            x_values,
            self.my_fantastic_logging["train_losses"][: epoch + 1],
            color="b",
            ls="-",
            label="loss_tr",
            linewidth=4,
        )
        ax.plot(
            x_values,
            self.my_fantastic_logging["val_losses"][: epoch + 1],
            color="r",
            ls="-",
            label="loss_val",
            linewidth=4,
        )
        ax2.plot(
            x_values,
            self.my_fantastic_logging["mean_fg_dice"][: epoch + 1],
            color="g",
            ls="dotted",
            label="pseudo dice",
            linewidth=3,
        )
        ax2.plot(
            x_values,
            self.my_fantastic_logging["ema_fg_dice"][: epoch + 1],
            color="g",
            ls="-",
            label="pseudo dice (mov. avg.)",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # Epoch times to check training speed consistency
        ax = ax_all[1]
        ax.plot(
            x_values,
            [
                i - j
                for i, j in zip(
                    self.my_fantastic_logging["epoch_end_timestamps"][: epoch + 1],
                    self.my_fantastic_logging["epoch_start_timestamps"],
                )
            ][: epoch + 1],
            color="b",
            ls="-",
            label="epoch duration",
            linewidth=4,
        )
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # Learning rate for encoder
        ax = ax_all[2]
        ax.plot(
            x_values,
            self.my_fantastic_logging["encoder_lrs"][: epoch + 1],
            color="blue",
            ls="-",
            label="Encoder learning rate",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        # Learning rate for decoder
        ax = ax_all[3]
        ax.plot(
            x_values,
            self.my_fantastic_logging["decoder_lrs"][: epoch + 1],
            color="red",
            ls="-",
            label="Decoder learning rate",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()


class nnUNetLoggerV2(nnUNetLogger):
    """
    This class is really trivial. Don't expect cool functionality here. This is my makeshift solution to problems
    arising from out-of-sync epoch numbers and numbers of logged loss values. It also simplifies the trainer class a
    little

    YOU MUST LOG EXACTLY ONE VALUE PER EPOCH FOR EACH OF THE LOGGING ITEMS! DONT FUCK IT UP
    """

    def __init__(self, verbose: bool = False):
        self.my_fantastic_logging = {
            "mean_fg_dice": list(),
            "ema_fg_dice": list(),
            "dice_per_class_or_region": list(),
            "train_losses": list(),
            "val_losses": list(),
            "encoder_lrs": list(),
            "decoder_lrs": list(),
            "seglayers_lrs": list(),
            "epoch_start_timestamps": list(),
            "epoch_end_timestamps": list(),
        }
        self.verbose = verbose
        # shut up, this logging is great

    def plot_progress_png(self, output_folder):
        # Infer the epoch from internal logging
        epoch = min([len(i) for i in self.my_fantastic_logging.values()]) - 1
        sns.set(font_scale=2.5)

        fig, ax_all = plt.subplots(
            5, 1, figsize=(30, 72)
        )  # Added one more subplot for additional LR

        # Regular progress.png as we are used to from previous nnU-Net versions
        ax = ax_all[0]
        ax2 = ax.twinx()
        x_values = list(range(epoch + 1))
        ax.plot(
            x_values,
            self.my_fantastic_logging["train_losses"][: epoch + 1],
            color="b",
            ls="-",
            label="loss_tr",
            linewidth=4,
        )
        ax.plot(
            x_values,
            self.my_fantastic_logging["val_losses"][: epoch + 1],
            color="r",
            ls="-",
            label="loss_val",
            linewidth=4,
        )
        ax2.plot(
            x_values,
            self.my_fantastic_logging["mean_fg_dice"][: epoch + 1],
            color="g",
            ls="dotted",
            label="pseudo dice",
            linewidth=3,
        )
        ax2.plot(
            x_values,
            self.my_fantastic_logging["ema_fg_dice"][: epoch + 1],
            color="g",
            ls="-",
            label="pseudo dice (mov. avg.)",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax2.set_ylabel("pseudo dice")
        ax.legend(loc=(0, 1))
        ax2.legend(loc=(0.2, 1))

        # Epoch times to check training speed consistency
        ax = ax_all[1]
        ax.plot(
            x_values,
            [
                i - j
                for i, j in zip(
                    self.my_fantastic_logging["epoch_end_timestamps"][: epoch + 1],
                    self.my_fantastic_logging["epoch_start_timestamps"],
                )
            ][: epoch + 1],
            color="b",
            ls="-",
            label="epoch duration",
            linewidth=4,
        )
        ylim = [0] + [ax.get_ylim()[1]]
        ax.set(ylim=ylim)
        ax.set_xlabel("epoch")
        ax.set_ylabel("time [s]")
        ax.legend(loc=(0, 1))

        # Learning rate for encoder
        ax = ax_all[2]
        ax.plot(
            x_values,
            self.my_fantastic_logging["encoder_lrs"][: epoch + 1],
            color="blue",
            ls="-",
            label="Encoder learning rate",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        # Learning rate for decoder
        ax = ax_all[3]
        ax.plot(
            x_values,
            self.my_fantastic_logging["decoder_lrs"][: epoch + 1],
            color="red",
            ls="-",
            label="Decoder learning rate",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        # Learning rate for seglayers
        ax = ax_all[4]
        ax.plot(
            x_values,
            self.my_fantastic_logging["seglayers_lrs"][: epoch + 1],
            color="red",
            ls="-",
            label="Segmentation Head learning rate",
            linewidth=4,
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("learning rate")
        ax.legend(loc=(0, 1))

        plt.tight_layout()

        fig.savefig(join(output_folder, "progress.png"))
        plt.close()
