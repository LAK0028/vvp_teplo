import os
import numpy as np
import json
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from natsort import natsorted


class ReadInputData:
    def __init__(self, simulation_folder_path):
        self.simulation_folder_path = simulation_folder_path

        self.kappa_path = None
        self.rho_c = None
        self.T_0 = None
        self.parameters = None

    def load_input_data(self):
        self.kappa_path = os.path.join(self.simulation_folder_path, "kappa.npy")
        self.rho_c_path = os.path.join(self.simulation_folder_path, "rho_c.npy")
        self.T_0_path = os.path.join(self.simulation_folder_path, "T_0.npy")
        self.parameters_path = os.path.join(self.simulation_folder_path, "parameters.json")

        self.kappa = np.load(self.kappa_path)
        self.rho_c = np.load(self.rho_c_path)
        self.T_0 = np.load(self.T_0_path)
        with open(self.parameters_path, 'r') as f:
            self.parameters = json.load(f)

        return self.kappa, self.rho_c, self.T_0, self.parameters


class Mesh:
    def __init__(self, num_cells_x, num_cells_y, dx, dy):
        self.num_cells_x = num_cells_x
        self.num_cells_y = num_cells_y
        self.dx = dx
        self.dy = dy

        self.a = num_cells_x * dx
        self.b = num_cells_y * dy

        self.x = np.linspace(0, self.a, num_cells_x + 1)
        self.y = np.linspace(0, self.b, num_cells_y + 1)

        self.X, self.Y = np.meshgrid(self.x, self.y)

    def get_grid_points(self):
        return self.X, self.Y

    def create_mesh_plot(self, matrix, title="title", bar_title="bar_title", min=None, max=None):
        fig, ax = plt.subplots(figsize=(9, 8))
        pc = ax.pcolormesh(self.X, self.Y, matrix, shading="flat",
                           cmap="inferno", vmin=min, vmax=max)

        ax.set_aspect('equal', adjustable='box')
        ax.set_title(title)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        cbar = fig.colorbar(pc, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label(bar_title)

        plt.tight_layout()
        return fig

    def show_result(self, matrix, title="title", bar_title="bar_title", min=None, max=None):
        self.create_mesh_plot(matrix, title, bar_title, min, max)
        plt.show()

    def save_result(self, matrix, title="title", bar_title="bar_title",
                    min=None, max=None, save_path="plot.png"):
        fig = self.create_mesh_plot(matrix, title, bar_title, min, max)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    def save_sequence_of_results(self, matrix_dict, title="title", bar_title="bar_title",
                                 min=None, max=None, save_path="results"):
        os.makedirs(save_path)

        for time_key in sorted(matrix_dict.keys()):
            matrix = matrix_dict[time_key]
            file_title = f"{title} {time_key}s"
            file_name = f"{save_path}/{time_key:06.0f}.png"

            self.save_result(matrix, title=file_title, bar_title=bar_title,
                             min=min, max=max, save_path=file_name)


def create_video_from_sequence(folder_path, output_path, fps=1):
    """
    Creates a video from images in a folder.

    Args:
        folder_path (str): Path to the folder containing image files.
        output_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the output video.

    Returns:
        None
    """
    files = natsorted(os.listdir(folder_path))

    images = []
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            images.append(imageio.imread(file_path))

    imageio.mimsave(output_path, images, fps=fps, codec="libx264")
