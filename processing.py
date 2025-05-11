import os
import numpy as np
import json


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
    def __init__(self, nx, ny, dx, dy):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy

        self.a = nx * dx
        self.b = ny * dy

        self.x = np.linspace(dx / 2, self.a - dx / 2, nx)
        self.y = np.linspace(dy / 2, self.b - dy / 2, ny)

        self.X, self.Y = np.meshgrid(self.x, self.y)

    def get_grid_points(self):
        return self.X, self.Y
