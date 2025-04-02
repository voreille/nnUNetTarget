import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.finetuning.custom_nnunet_logger import (
    nnUNetLoggerV1,
    nnUNetLoggerV2,
)
from nnunetv2.training.nnUNetTrainer.variants.finetuning.scheduler import (
    PolyLRScheduler,
)


class nnUNetTrainerFineTuning(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.freeze_encoder_epochs = 0
        self.encoder_lr = 1e-2
        self.decoder_lr = 1e-2
        self.encoder_frozen = True  # Track whether encoder is frozen
        self.logger = nnUNetLoggerV1()
        # self.num_epochs = 3
        # hard-coded paths for pretrained weights..
        self.pretrained_encoder_path = (
            "/home/vincent/repos/ssl-bm/weights/cnn3d_nnunet_2ndgood.pt"
        )

    def configure_optimizers(self):
        """
        Configure optimizer and Poly learning rate scheduler for fine-tuning with SGD.
        """
        encoder_params = []
        decoder_params = []

        for name, param in self.network.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
                if self.encoder_frozen:
                    param.requires_grad = False
            else:
                decoder_params.append(param)

        optimizer = torch.optim.SGD(
            [
                {"params": encoder_params, "lr": self.encoder_lr},
                {"params": decoder_params, "lr": self.decoder_lr},
            ],
            momentum=0.99,
            nesterov=True,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = PolyLRScheduler(optimizer, self.num_epochs)
        return optimizer, lr_scheduler

    def initialize(self):
        super().initialize()
        self.load_pretrained_weights()

        if self.encoder_frozen:
            self._freeze_encoder()

    def load_pretrained_weights(self):
        state_dict = torch.load(self.pretrained_encoder_path)
        encoder_state_dict = {
            k[len("encoder.") :]: v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        self.network.encoder.load_state_dict(encoder_state_dict)

        # Check that encoder and decoder encoder have the same weights
        encoder_params = {k: v for k, v in self.network.encoder.state_dict().items()}
        decoder_encoder_params = {
            k: v for k, v in self.network.decoder.encoder.state_dict().items()
        }

        for key in encoder_params:
            assert torch.equal(encoder_params[key], decoder_encoder_params[key]), (
                f"Mismatch in {key}"
            )

    def on_train_epoch_start(self):
        """
        Log separate learning rates for encoder and decoder.
        """
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        encoder_lr = self.optimizer.param_groups[0]["lr"]
        decoder_lr = self.optimizer.param_groups[1]["lr"]
        self.print_to_log_file(f"Encoder learning rate: {encoder_lr:.5f}")
        self.print_to_log_file(f"Decoder learning rate: {decoder_lr:.5f}")
        self.logger.log("encoder_lrs", encoder_lr, self.current_epoch)
        self.logger.log("decoder_lrs", decoder_lr, self.current_epoch)

    def run_training(self):
        """
        Overridden training loop with staged fine-tuning logic.
        """

        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            if self.encoder_frozen and epoch >= self.freeze_encoder_epochs:
                self._unfreeze_encoder()
                self.encoder_frozen = False
                # self.optimizer, self.lr_scheduler = self.configure_optimizers()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

    def _freeze_encoder(self):
        self.print_to_log_file("Freezing encoder parameters...")
        for name, param in self.network.named_parameters():
            if "encoder" in name:
                param.requires_grad = False

    def _unfreeze_encoder(self):
        self.print_to_log_file("Unfreezing encoder parameters...")
        for name, param in self.network.named_parameters():
            if "encoder" in name:
                param.requires_grad = True


class nnUNetTrainerFineTuningV2(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.freeze_encoder_epochs = 0
        self.encoder_lr = 1e-2
        self.decoder_lr = 1e-2
        self.seglayers_lr = 1e-2
        self.encoder_frozen = True  # Track whether encoder is frozen
        self.decoder_frozen = True  # Track whether encoder is frozen
        self.logger = nnUNetLoggerV2()
        # self.num_epochs = 3
        # hard-coded paths for pretrained weights..
        self.pretrained_encoder_path = "/home/vincent/repos/ssl-bm/weights/cnn3d_nnunet_local_global_100ep_checkpoint.pth"

    def initialize(self):
        super().initialize()
        self.load_pretrained_weights()

        if self.encoder_frozen:
            self._freeze_encoder()

        if self.decoder_frozen:
            self._freeze_decoder()

    def configure_optimizers(self):
        """
        Configure optimizer and Poly learning rate scheduler for fine-tuning with SGD.
        """
        encoder_params = []
        decoder_params = []
        seglayers_params = []

        for name, param in self.network.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
                if self.encoder_frozen:
                    param.requires_grad = False
            elif (
                "decoder" in name and "encoder" not in name and "seg_layers" not in name
            ):
                decoder_params.append(param)
            elif "seg_layers" in name:
                seglayers_params.append(param)
            else:
                raise ValueError(
                    "MMMMMhh there is an implementation "
                    "when filtering parameter for optimizers"
                )

        optimizer = torch.optim.SGD(
            [
                {"params": encoder_params, "lr": self.encoder_lr},
                {"params": decoder_params, "lr": self.decoder_lr},
                {"params": seglayers_params, "lr": self.seglayers_lr},
            ],
            momentum=0.99,
            nesterov=True,
            weight_decay=self.weight_decay,
        )

        lr_scheduler = PolyLRScheduler(optimizer, self.num_epochs)
        return optimizer, lr_scheduler

    def load_pretrained_weights(self):
        self.print_to_log_file(
            f"Loading pretrained weights {self.pretrained_encoder_path}"
        )
        state_dict = torch.load(self.pretrained_encoder_path, weights_only=False)
        # Filter encoder_q out
        network_state_dict = {
            key[len("encoder_q.") :]: item
            for key, item in state_dict["model_state_dict"].items()
            if key.startswith("encoder_q.")
        }
        # Take only model, see Vincent's implementation
        network_state_dict = {
            key[len("model.") :]: item
            for key, item in network_state_dict.items()
            if key.startswith("model.")
        }
        # Filter the seg_layers out since we want to train them
        network_state_dict = {
            k: v
            for k, v in network_state_dict.items()
            if not k.startswith("decoder.seg_layers")
        }
        load_info = self.network.load_state_dict(network_state_dict, strict=False)
        self.print_to_log_file("Missing keys:", load_info.missing_keys)
        self.print_to_log_file("Unexpected keys:", load_info.unexpected_keys)

    def _freeze_encoder(self):
        self.print_to_log_file("Freezing encoder parameters...")
        for name, param in self.network.named_parameters():
            if "encoder" in name:
                param.requires_grad = False

    def _unfreeze_encoder(self):
        self.print_to_log_file("Unfreezing encoder parameters...")
        for name, param in self.network.named_parameters():
            if "encoder" in name:
                param.requires_grad = True

    def _freeze_decoder(self):
        self.print_to_log_file("Freezing encoder parameters...")
        for name, param in self.network.named_parameters():
            if "decoder" in name and "encoder" not in name:
                param.requires_grad = False

    def _unfreeze_encoder(self):
        self.print_to_log_file("Unfreezing encoder parameters...")
        for name, param in self.network.named_parameters():
            if "decoder" in name and "encoder" not in name:
                param.requires_grad = True

    def _freeze_seglayers(self):
        self.print_to_log_file("Freezing encoder parameters...")
        raise NotImplementedError("TO FINISH!!!")
        for name, param in self.network.named_parameters():
            if "seg_layers" in name:
                param.requires_grad = False

    def _unfreeze_seglayers(self):
        self.print_to_log_file("Unfreezing encoder parameters...")
        raise NotImplementedError("TO FINISH!!!")
        for name, param in self.network.named_parameters():
            if "seg_layers" in name:
                param.requires_grad = True

    def on_train_epoch_start(self):
        """
        Log separate learning rates for encoder and decoder.
        """
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file("")
        self.print_to_log_file(f"Epoch {self.current_epoch}")
        encoder_lr = self.optimizer.param_groups[0]["lr"]
        decoder_lr = self.optimizer.param_groups[1]["lr"]
        self.print_to_log_file(f"Encoder learning rate: {encoder_lr:.5f}")
        self.print_to_log_file(f"Decoder learning rate: {decoder_lr:.5f}")
        self.logger.log("encoder_lrs", encoder_lr, self.current_epoch)
        self.logger.log("decoder_lrs", decoder_lr, self.current_epoch)

    def run_training(self):
        """
        Overridden training loop with staged fine-tuning logic.
        """

        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            if self.encoder_frozen and epoch >= self.freeze_encoder_epochs:
                self._unfreeze_encoder()
                self.encoder_frozen = False
                # self.optimizer, self.lr_scheduler = self.configure_optimizers()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()
