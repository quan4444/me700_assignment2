import numpy as np
from a2 import math_utils as mut
import pytest


def test_local_elastic_stiffness_matrix_3D_beam():
    E = 210e9
    nu = 0.3
    A = 0.01
    L = 2.0
    Iy = 1e-5
    Iz = 1e-5
    J = 1e-6
    k_e = mut.local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
    assert k_e.shape == (12, 12)  # Check shape
    assert np.allclose(k_e, k_e.T)  # Check symmetry


def test_check_unit_vector():
    vec = np.array([1, 0, 0])
    mut.check_unit_vector(vec)  # Should not raise an error

    with pytest.raises(ValueError):
        mut.check_unit_vector(np.array([1, 1, 1]))  # Not a unit vector


def test_check_parallel():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    mut.check_parallel(vec1, vec2)  # Should not raise an error

    with pytest.raises(ValueError):
        mut.check_parallel(vec1, np.array([2, 0, 0]))  # Parallel vectors


def test_rotation_matrix_3D():
    x1, y1, z1 = 0, 0, 0
    x2, y2, z2 = 1, 0, 0
    gamma = mut.rotation_matrix_3D(x1, y1, z1, x2, y2, z2)
    assert gamma.shape == (3, 3)  # Check shape
    assert np.allclose(gamma @ gamma.T, np.eye(3))  # Check orthogonality


def test_transformation_matrix_3D():
    gamma = np.eye(3)
    Gamma = mut.transformation_matrix_3D(gamma)
    assert Gamma.shape == (12, 12)  # Check shape
    assert np.allclose(Gamma[:3, :3], gamma)  # Check block diagonal

# Test for local_geometric_stiffness_matrix_3D_beam
def test_local_geometric_stiffness_matrix_3D_beam():
    A = 0.01
    L = 2.0
    Fx2 = 1000
    Mx2 = 500
    My1 = 200
    Mz1 = 300
    My2 = 400
    Mz2 = 500
    I_rho = 1e-6
    k_g = mut.local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert k_g.shape == (12, 12)  # Check shape
    assert np.allclose(k_g, k_g.T)  # Check symmetry


def test_local_geometric_stiffness_matrix_3D_beam_without_interaction_terms():
    A = 0.01
    L = 2.0
    Fx2 = 1000
    I_rho = 1e-6
    k_g = mut.local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(L, A, I_rho, Fx2)
    assert k_g.shape == (12, 12)  # Check shape
    assert np.allclose(k_g, k_g.T)  # Check symmetry