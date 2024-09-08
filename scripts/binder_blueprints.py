import os

import torch
import numpy as np


class BinderBlueprints:
    ELEMENT_TYPES = {"H": 0, "E": 1, "C": 2}

    def __init__(
        self,
        elements: str = "HHH",
        size: int = 80,
        linker: int = 3,
        element_lengths: list = None,
        linker_lengths: list = None,
        adj: np.ndarray = None,
        sse: np.ndarray = None,
    ):
        """
        Generate a binder scaffold blueprint with the specified secondary structure elements and size.

        :param elements: A string like "HHH" indicating secondary structure elements (e.g., H-helices, E-strands).
        :param size: Total length of the scaffold.
        :param linker: Default length of the linker between secondary structure elements.
        :param element_lengths: A list of lengths for each secondary structure element.
        :param linker_lengths: A list of lengths for each linker.
        :param adj: Precomputed adjacency matrix, if provided.
        :param sse: Precomputed secondary structure matrix, if provided.
        """
        self.elements = elements
        self.size = size
        self.linker = linker
        self.element_lengths = element_lengths or []
        self.linker_lengths = linker_lengths or []

        if adj is not None and sse is not None:
            self.adj = adj
            self.sse = sse
        else:
            self.calculate_structure_lengths()
            self.sse, self.adj = self.generate_matrices()

    @property
    def num_elements(self) -> int:
        """Returns the number of secondary structure elements."""
        return len(self.elements)

    @property
    def sse_lengths(self) -> list:
        """Returns the lengths of secondary structure elements."""
        return self.element_lengths

    @property
    def linker_lengths(self) -> list:
        """Returns the lengths of linkers."""
        return self._linker_lengths

    @linker_lengths.setter
    def linker_lengths(self, lengths: list):
        if lengths and len(lengths) != self.num_elements - 1:
            raise ValueError("linker_lengths must have length num_elements - 1.")
        self._linker_lengths = lengths or [self.linker] * (self.num_elements - 1)

    @classmethod
    def from_files(cls, adj_file: str, sse_file: str):
        """Create an instance of StructureMatrixGenerator by loading matrices from files."""
        adj = torch.load(adj_file).numpy()
        sse = torch.load(sse_file).numpy()
        return cls(adj=adj, sse=sse)

    def get_adj(self) -> np.ndarray:
        """Returns the adjacency matrix."""
        return self.adj

    def plot_adj(self):
        """Plot the adjacency matrix."""
        import matplotlib.pyplot as plt

        plt.imshow(self.adj, cmap="viridis", origin="lower")
        plt.colorbar(label="Value")
        plt.title("Adjacency Matrix")
        plt.show()

    def get_sse(self) -> np.ndarray:
        """Returns the secondary structure elements."""
        return self.sse

    def calculate_structure_lengths(self):
        """
        Calculate the length of each secondary structure element and linkers
        based on the total size or specified lengths. Normally, the lengths of
        strands are half of helices.
        """
        num_elements = len(self.elements)
        # Check if specific lengths are provided, otherwise calculate based on the total size
        if self.element_lengths:
            self.element_lengths = self.element_lengths
        else:
            self.element_lengths = [0] * num_elements
            total_linker_length = (num_elements - 1) * self.linker
            total_sse_length = self.size - total_linker_length

            num_h = self.elements.count("H")
            num_e = self.elements.count("E")

            if num_h > 0 and num_e > 0:
                length_per_h = total_sse_length // (num_h + num_e / 2)
                length_per_h = int(length_per_h)
                length_per_e = int(length_per_h // 2)
            elif num_h > 0:
                length_per_h = int(total_sse_length // num_h)
                length_per_e = 0
            elif num_e > 0:
                length_per_e = int(total_sse_length // num_e)
                length_per_h = 0
            else:
                length_per_h = length_per_e = 0

            for idx, sse in enumerate(self.elements):
                if sse == "H":
                    self.element_lengths[idx] = length_per_h
                elif sse == "E":
                    self.element_lengths[idx] = length_per_e

            used_length = sum(self.element_lengths) + (num_elements - 1) * self.linker
            extra_length = self.size - used_length

            for i in range(num_elements):
                if extra_length <= 0:
                    break
                if self.elements[i] == "H" or self.elements[i] == "E":
                    self.element_lengths[i] += 1
                    extra_length -= 1

        # Set linker lengths
        if self.linker_lengths:
            self.linker_lengths = self.linker_lengths
        else:
            self.linker_lengths = [self.linker] * (num_elements - 1)

    def generate_matrices(self):
        """Generate the secondary structure and adjacency matrices based on elements and size."""
        total_length = self.size
        ssem = np.full((total_length,), self.ELEMENT_TYPES["C"])
        adjm = np.full((total_length, total_length), 2)

        current_position = 0
        for idx, sse in enumerate(self.elements):
            sse_length = self.element_lengths[idx]
            sse_type = self.ELEMENT_TYPES[sse]
            ssem[int(current_position) : int(current_position + sse_length)] = sse_type
            adjm[
                int(current_position) : int(current_position + sse_length),
                int(current_position) : int(current_position + sse_length),
            ] = 0
            current_position += sse_length
            if idx < self.num_elements - 1:
                current_position += self.linker_lengths[idx]

        # Set adjacency for interactions between SSEs
        for i in range(self.num_elements):
            if self.elements[i] in ["H", "E"]:
                for j in range(self.num_elements):
                    if i != j and self.elements[j] in ["H", "E"]:
                        sse_i_start = int(
                            sum(self.element_lengths[:i]) + sum(self.linker_lengths[:i])
                        )
                        sse_i_end = int(sse_i_start + self.element_lengths[i])
                        sse_j_start = int(
                            sum(self.element_lengths[:j]) + sum(self.linker_lengths[:j])
                        )
                        sse_j_end = int(sse_j_start + self.element_lengths[j])

                        adjm[sse_i_start:sse_i_end, sse_j_start:sse_j_end] = 1
                        adjm[sse_j_start:sse_j_end, sse_i_start:sse_i_end] = 1

        return ssem, adjm

    def save_adj_sse(self, output_dir: str = "outputs"):
        """Save the ADJ and SSE matrices to files in the specified directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        element_lengths = "_".join(map(str, self.element_lengths))
        file_prefix = f"{self.elements}_{self.size}_el{element_lengths}_li{self.linker}"
        sse_file = os.path.join(output_dir, f"{file_prefix}_ss.pt")
        adj_file = os.path.join(output_dir, f"{file_prefix}_adj.pt")
        torch.save(torch.from_numpy(self.sse).float(), sse_file)
        torch.save(torch.from_numpy(self.adj).float(), adj_file)

    def __repr__(self):
        return (
            f"BinderBlueprints(elements={self.elements}, size={self.size}, "
            f"linker={self.linker}, element_lengths={self.element_lengths}, "
            f"linker_lengths={self.linker_lengths})"
        )
