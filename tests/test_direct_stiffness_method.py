import numpy as np
import pytest
from a2 import direct_stiffness_method as dsm
from a2 import math_utils as mut


@pytest.fixture
def OneLine():
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = np.array([[0, 1]])
    return dsm.Mesh(nodes,elements)


@pytest.fixture
def OneLineMat():
    onelinemat=dsm.MaterialParams()
    onelinemat.add_material(
        subdomain_id=1,E=210e9, nu=0.3, A=0.01, I_z=1e-5, I_y=1e-5, I_p=1e-5, J=1e-5, local_z_axis=[0, 0, 1]
    )
    onelinemat.assign_subdomain(subdomain_id=1,element_ids=[0])
    return onelinemat


@pytest.fixture
def OneLineBCs():
    bcs=dsm.BoundaryConditions()
    bcs.add_support(0, u_x=0, u_y=0, u_z=0, theta_x=0, theta_y=0, theta_z=0)
    bcs.add_load(1, Fx=0, Fy=-1000, Fz=0, Mx=0, My=0, Mz=0)
    return bcs


@pytest.fixture
def knownKGlobal():
    E,nu,A,Iz,Iy,Ip,J,L=210e9,0.3,0.01,1e-5,1e-5,1e-5,1e-5,1.0
    known = np.zeros((12, 12))
    # Axial terms - extension of local x axis
    axial_stiffness = E * A / L
    known[0, 0] = axial_stiffness
    known[0, 6] = -axial_stiffness
    known[6, 0] = -axial_stiffness
    known[6, 6] = axial_stiffness
    # Torsion terms - rotation about local x axis
    torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
    known[3, 3] = torsional_stiffness
    known[3, 9] = -torsional_stiffness
    known[9, 3] = -torsional_stiffness
    known[9, 9] = torsional_stiffness
    # Bending terms - bending about local z axis
    known[1, 1] = E * 12.0 * Iz / L ** 3.0
    known[1, 7] = E * -12.0 * Iz / L ** 3.0
    known[7, 1] = E * -12.0 * Iz / L ** 3.0
    known[7, 7] = E * 12.0 * Iz / L ** 3.0
    known[1, 5] = E * 6.0 * Iz / L ** 2.0
    known[5, 1] = E * 6.0 * Iz / L ** 2.0
    known[1, 11] = E * 6.0 * Iz / L ** 2.0
    known[11, 1] = E * 6.0 * Iz / L ** 2.0
    known[5, 7] = E * -6.0 * Iz / L ** 2.0
    known[7, 5] = E * -6.0 * Iz / L ** 2.0
    known[7, 11] = E * -6.0 * Iz / L ** 2.0
    known[11, 7] = E * -6.0 * Iz / L ** 2.0
    known[5, 5] = E * 4.0 * Iz / L
    known[11, 11] = E * 4.0 * Iz / L
    known[5, 11] = E * 2.0 * Iz / L
    known[11, 5] = E * 2.0 * Iz / L
    # Bending terms - bending about local y axis
    known[2, 2] = E * 12.0 * Iy / L ** 3.0
    known[2, 8] = E * -12.0 * Iy / L ** 3.0
    known[8, 2] = E * -12.0 * Iy / L ** 3.0
    known[8, 8] = E * 12.0 * Iy / L ** 3.0
    known[2, 4] = E * -6.0 * Iy / L ** 2.0
    known[4, 2] = E * -6.0 * Iy / L ** 2.0
    known[2, 10] = E * -6.0 * Iy / L ** 2.0
    known[10, 2] = E * -6.0 * Iy / L ** 2.0
    known[4, 8] = E * 6.0 * Iy / L ** 2.0
    known[8, 4] = E * 6.0 * Iy / L ** 2.0
    known[8, 10] = E * 6.0 * Iy / L ** 2.0
    known[10, 8] = E * 6.0 * Iy / L ** 2.0
    known[4, 4] = E * 4.0 * Iy / L
    known[10, 10] = E * 4.0 * Iy / L
    known[4, 10] = E * 2.0 * Iy / L
    known[10, 4] = E * 2.0 * Iy / L
    return known


def test_mesh():
    
    with pytest.raises(Exception):
        nodes = np.array([
            [0,1],
            [1,1]
        ])
        elements=np.array([[0,1]])
        dsm.Mesh(nodes,elements)
    
    with pytest.raises(Exception):
        nodes = np.array([
            [0,1,0],
            [0,0,0]
        ])
        elements=np.array([[1]])
        dsm.Mesh(nodes,elements)


def test_matparams():
    mp = dsm.MaterialParams()
    mp.add_material(
        subdomain_id=1,E=210e9, nu=0.3, A=0.01, I_z=1e-5, I_y=1e-5, I_p=1e-5, J=1e-5, local_z_axis=[0, 0, 1]
    )
    mp.assign_subdomain(subdomain_id=1,element_ids=[0])

    with pytest.raises(Exception):
        mp.assign_subdomain(subdomain_id=4,element_ids=[0])


def test_assemble_global_stiffness_matrix(OneLine,OneLineMat,knownKGlobal):
    K_global=dsm.assemble_global_stiffness_matrix(OneLine,OneLineMat)
    assert np.allclose(knownKGlobal,K_global)


def test_solve_system(OneLine,OneLineMat,OneLineBCs):
    K_global=dsm.assemble_global_stiffness_matrix(OneLine,OneLineMat)
    disps,rxns=dsm.solve_system(K_global,OneLineBCs)
    known_disps = [0,-0.00015873,0,0,0,0.0002381]
    known_rxns = [0,1000,0,0,0,-1000]
    assert np.allclose(disps[1,:],known_disps)
    assert np.allclose(rxns[0,:],known_rxns)


def test_generate_mesh_and_solve():
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = np.array([[0, 1]])
    subdomain_dict = {1:[210e9,0.3,0.01,1e-5,1e-5,1e-5,1e-5,[0,0,1]]}
    subdomain_elements = {1:[0]}
    list_fixed_nodes_id = [0]
    list_pinned_nodes_id = []
    load_dict = {1:[0,-1000,0,0,0,0]}

    disps,rxns = dsm.generate_mesh_and_solve(
        nodes,elements,subdomain_dict,subdomain_elements,list_fixed_nodes_id,list_pinned_nodes_id,load_dict
    )

    known_disps = [0,-0.00015873,0,0,0,0.0002381]
    known_rxns = [0,1000,0,0,0,-1000]

    assert np.allclose(disps[1,:],known_disps)
    assert np.allclose(rxns[0,:],known_rxns)