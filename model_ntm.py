"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from argparse import ArgumentParser

from ntm.ntm import NTM
from ntm.controller import LSTMController
from ntm.head import NTMReadHead, NTMWriteHead
from ntm.memory import NTMMemory


class MyNTM(nn.Module):
    def __init__(self,
                 num_inputs: int = 9,
                 num_outputs: int = 8,
                 controller_size: int = 100,
                 controller_layers: int = 1,
                 num_heads: int = 1,
                 memory_n: int = 128,
                 memory_m: int = 20,
                 ):
        """Initialize an EncapsulatedNTM.

        :param num_inputs: External number of inputs.
        :param num_outputs: External number of outputs.
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param memory_n: Number of rows in the memory bank.
        :param memory_m: Number of cols/features in the memory bank.
        """
        super(MyNTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.memory_n = memory_n
        self.memory_m = memory_m

        # Create the NTM components
        memory = NTMMemory(memory_n, memory_m)
        controller = LSTMController(num_inputs + memory_m * num_heads, controller_size, controller_layers)
        heads = nn.ModuleList([])
        for i in range(num_heads):
            heads += [
                NTMReadHead(memory, controller_size),
                NTMWriteHead(memory, controller_size)
            ]

        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads)
        self.memory = memory

    def init_sequence(self, batch_size, device):
        self.device = device
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size, device)
        self.previous_state = self.ntm.create_new_state(batch_size, device)

    def forward(self, x=None):
        if x is None:
            x = torch.zeros(self.batch_size, self.num_inputs).to(self.device)

        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ntm_controller_size', type=int, default=100)
        parser.add_argument('--ntm_controller_layers', type=int, default=1)
        parser.add_argument('--ntm_num_heads', type=int, default=1)
        parser.add_argument('--ntm_memory_n', type=int, default=128)
        parser.add_argument('--ntm_memory_m', type=int, default=20)
        return parser