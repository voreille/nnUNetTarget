from torch.optim.lr_scheduler import _LRScheduler


class PolyLRScheduler(_LRScheduler):

    def __init__(self,
                 optimizer,
                 max_steps: int,
                 exponent: float = 0.9,
                 current_step: int = None):
        """
        Polynomial decay scheduler that respects the initial_lr for each parameter group.

        Args:
            optimizer: The optimizer with parameter groups.
            max_steps: Total number of steps (epochs or iterations).
            exponent: Power of the polynomial decay.
            current_step: Current step, defaults to -1.
        """
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer,
                         current_step if current_step is not None else -1,
                         False)

    def step(self, current_step=None):
        """
        Update the learning rates for all parameter groups based on the current step.

        Args:
            current_step: Optional, manually specify the current step.
        """
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        # Calculate new learning rate for each parameter group
        for param_group in self.optimizer.param_groups:
            initial_lr = param_group.get('initial_lr', param_group['lr'])
            param_group['lr'] = initial_lr * (
                1 - current_step / self.max_steps)**self.exponent
