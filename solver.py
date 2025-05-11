from scipy.sparse import lil_matrix, csr_matrix
import numpy as np


def conduction_solving_step(
    kappa: np.ndarray,
    rho_c: np.ndarray,
    T_0: np.ndarray,
    num_cells_x: int,
    num_cells_y: int,
    dx: float,
    dy: float,
    dt: float
) -> np.ndarray:
    """Performs one time step of a 2D transient heat conduction simulation using finite difference method.

    This function calculates the updated temperature distribution in a 2D grid based on thermal conductivity,
    volumetric heat capacity, and the current temperature distribution.

    Args:
        kappa (np.ndarray): 2D array of thermal conductivity values [W/(m·K)].
        rho_c (np.ndarray): 2D array of volumetric heat capacities (density · specific heat) [J/(m^3·K)].
        T_0 (np.ndarray): 2D array of initial temperatures [K].
        num_cells_x (int): Number of cells along the x-axis.
        num_cells_y (int): Number of cells along the y-axis.
        dx (float): Grid spacing in the x-direction [m].
        dy (float): Grid spacing in the y-direction [m].
        dt (float): Time step size [s].

    Returns:
        np.ndarray: 2D array of updated temperatures after one time step [K].
    """
    T = T_0.copy()
    updated_T = T.copy()

    for i in range(num_cells_x):
        for j in range(num_cells_y):
            if i > 0:
                dT_left = (T[i - 1, j] - T[i, j]) / (((1 / kappa[i, j]) + (1 / kappa[i - 1, j])) * dx ** 2)
            else:
                dT_left = 0

            if j > 0:
                dT_under = (T[i, j - 1] - T[i, j]) / (((1 / kappa[i, j]) + (1 / kappa[i, j - 1])) * dy ** 2)
            else:
                dT_under = 0

            if i < num_cells_x - 1:
                dT_right = (T[i + 1, j] - T[i, j]) / (((1 / kappa[i, j]) + (1 / kappa[i + 1, j])) * dx ** 2)
            else:
                dT_right = 0

            if j < num_cells_y - 1:
                dT_upper = (T[i, j + 1] - T[i, j]) / (((1 / kappa[i, j]) + (1 / kappa[i, j + 1])) * dy ** 2)
            else:
                dT_upper = 0

            updated_T[i, j] = T[i, j] + (2 * dt / rho_c[i, j]) * (dT_left + dT_under + dT_right + dT_upper)

    return updated_T


def loop_building_conduction_sparse_matrix(
    kappa: np.ndarray,
    rho_c: np.ndarray,
    num_cells_x: int,
    num_cells_y: int,
    dx: float,
    dy: float,
) -> csr_matrix:
    """
    Constructs a sparse matrix for 2D heat conduction using finite difference method.

    This function builds a sparse matrix representing heat transfer between cells in a 2D grid.
    Thermal conductivity and volumetric heat capacity can vary per cell.

    Args:
        kappa (np.ndarray): 2D array of thermal conductivity values [W/(m·K)].
        rho_c (np.ndarray): 2D array of volumetric heat capacities (density · specific heat) [J/(m^3·K)].
        num_cells_x (int): Number of cells along the x-axis.
        num_cells_y (int): Number of cells along the y-axis.
        dx (float): Grid spacing in the x-direction [m].
        dy (float): Grid spacing in the y-direction [m].

    Returns:
        csr_matrix: A sparse matrix in Compressed Sparse Row (CSR) format, used in the heat equation.
    """
    A = lil_matrix((num_cells_x * num_cells_y, num_cells_x * num_cells_y))

    def index_2D_to_1D(i, j):
        return i * num_cells_y + j

    for i in range(num_cells_x):
        for j in range(num_cells_y):
            if i > 0:
                kappa_left = 1 / (((1 / kappa[i, j]) + (1 / kappa[i - 1, j])) * dx ** 2)
                A[index_2D_to_1D(i, j), index_2D_to_1D(i - 1, j)] = (2 / (rho_c[i, j])) * kappa_left
            else:
                kappa_left = 0

            if j > 0:
                kappa_under = 1 / (((1 / kappa[i, j]) + (1 / kappa[i, j - 1])) * dy ** 2)
                A[index_2D_to_1D(i, j), index_2D_to_1D(i, j - 1)] = (2 / (rho_c[i, j])) * kappa_under
            else:
                kappa_under = 0

            if i < num_cells_x - 1:
                kappa_right = 1 / (((1 / kappa[i, j]) + (1 / kappa[i + 1, j])) * dx ** 2)
                A[index_2D_to_1D(i, j), index_2D_to_1D(i + 1, j)] = (2 / (rho_c[i, j])) * kappa_right
            else:
                kappa_right = 0

            if j < num_cells_y - 1:
                kappa_upper = 1 / (((1 / kappa[i, j]) + (1 / kappa[i, j + 1])) * dy ** 2)
                A[index_2D_to_1D(i, j), index_2D_to_1D(i, j + 1)] = (2 / (rho_c[i, j])) * kappa_upper
            else:
                kappa_upper = 0

            A[index_2D_to_1D(i, j), index_2D_to_1D(i, j)] = -(2 / (rho_c[i, j])) * \
                (kappa_left + kappa_under + kappa_right + kappa_upper)

    return A.tocsr()


def conduction_solving_step_matrix(
    A: csr_matrix,
    T_0: np.ndarray,
    dt: float
) -> np.ndarray:
    """
    Performs one time step of a 2D heat conduction simulation using matrix multiplication
    with a precomputed sparse matrix A.

    Args:
        A (csr_matrix): A sparse matrix in Compressed Sparse Row (CSR) format, used in the heat equation.
        T_0 (np.ndarray): 2D array of initial temperatures [K].
        dt (float): Time step size [s].

    Returns:
        np.ndarray: 2D array of updated temperatures after one time step [K].
    """
    T_0_flat = T_0.flatten()

    updated_T_flat = T_0_flat + dt * A @ T_0_flat
    updated_T = updated_T_flat.reshape(T_0.shape)

    return updated_T


def transient_conduction(
    kappa: np.ndarray,
    rho_c: np.ndarray,
    T_0: np.ndarray,
    num_cells_x: int,
    num_cells_y: int,
    dx: float,
    dy: float,
    dt: float,
    solver: str,
    tot_time: float,
    save_time: float
) -> list[np.ndarray]:
    """
    Simulates transient 2D heat conduction simulation using either element-wise or matrix-based solver.

    This function computes the temperature distribution in a 2D grid over time using a finite difference method.
    It supports two solvers: direct element-wise and sparse matrix multiplication using a precomputed sparse matrix.

    Args:
        kappa (np.ndarray): 2D array of thermal conductivity values [W/(m·K)].
        rho_c (np.ndarray): 2D array of volumetric heat capacities (density · specific heat) [J/(m^3·K)].
        T_0 (np.ndarray): 2D array of initial temperatures [K].
        num_cells_x (int): Number of cells along the x-axis.
        num_cells_y (int): Number of cells along the y-axis.
        dx (float): Grid spacing in the x-direction [m].
        dy (float): Grid spacing in the y-direction [m].
        dt (float): Time step size [s].
        solver (str): Solver type "element_wise" or "matrix_multiplication".
        tot_time (float): Total simulation time [s].
        save_time (float): Results saving interval [s].

    Returns:
        List[np.ndarray]: List of 2D arrays representing temperature distributions at each saved time step.

    Raises:
        ValueError: If the provided solver string is not recognized.
    """
    n_iterations = int(tot_time / dt)
    save_every_steps = int(save_time / dt)
    T = T_0.copy()
    T_history: list[np.ndarray] = []

    if solver == "element_wise":
        for step in range(n_iterations):
            T = conduction_solving_step(kappa, rho_c, T, num_cells_x, num_cells_y, dx, dy, dt)
            if step % save_every_steps == 0:
                T_history.append(T.copy())
    elif solver == "matrix_multiplication":
        A = loop_building_conduction_sparse_matrix(kappa, rho_c, num_cells_x, num_cells_y, dx, dy)
        for step in range(n_iterations):
            T = conduction_solving_step_matrix(A, T_0, dt)
            if step % save_every_steps == 0:
                T_history.append(T.copy())
    else:
        raise ValueError(f'Unknown solver "{solver}". Use "element_wise" or "matrix_multiplication."')

    return T_history
