import numpy as np
from a2 import math_utils as mut
from typing import List, Dict, Tuple

class Mesh:
    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        """
        Initialize the mesh with nodes and elements.
        
        :param nodes: NumPy array of shape (n_nodes, 3) representing node coordinates.
        :param elements: NumPy array of shape (n_elements, 2) representing element connectivity.
        """
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate the inputs for nodes and elements."""
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("Nodes must be a 2D array with shape (n_nodes, 3).")
        if self.elements.ndim != 2 or self.elements.shape[1] != 2:
            raise ValueError("Elements must be a 2D array with shape (n_elements, 2).")
        
        # Validate that all node IDs in elements exist in the nodes array
        max_node_id = len(self.nodes) - 1
        for element in self.elements:
            for node_id in element:
                if node_id < 0 or node_id > max_node_id:
                    raise ValueError(f"Node ID {node_id} in elements does not exist in the nodes array.")
        

class MaterialParams:
    def __init__(self, subdomain_dict: Dict[int, List[float]]):
        """
        Initialize material properties with a dictionary of subdomains and their properties.
        
        :param subdomain_dict: Dictionary where keys are subdomain IDs and values are lists of material properties.
                               Example: {1: [E, nu, A, I_z, I_y, I_p, J, local_z_axis]}
        """
        self.materials = {}
        self.subdomains = {}
        self._initialize_materials(subdomain_dict)

    def _initialize_materials(self, subdomain_dict: Dict[int, List[float]]):
        """Add material properties for each subdomain."""
        for subdomain_id, mat_params in subdomain_dict.items():
            self.add_material(subdomain_id, *mat_params)

    def add_material(self, subdomain_id: int, E: float, nu: float, A: float, I_z: float, I_y: float, I_p: float, J: float, local_z_axis=None):
        """
        Add material properties for a subdomain.
        
        :param subdomain_id: ID of the subdomain.
        :param E: Young's modulus.
        :param nu: Poisson's ratio.
        :param A: Cross-sectional area.
        :param I_z: Moment of inertia about the z-axis.
        :param I_y: Moment of inertia about the y-axis.
        :param I_p: Polar moment of inertia.
        :param J: Torsional constant.
        :param local_z_axis: Local z-axis orientation (optional).
        """
        self._validate_material_properties(E, nu, A, I_z, I_y, I_p, J)
        self.materials[subdomain_id] = {
            'E': E, 'nu': nu, 'A': A, 'I_z': I_z, 'I_y': I_y, 'I_p': I_p, 'J': J, 'local_z_axis': local_z_axis
        }

    def _validate_material_properties(self, E: float, nu: float, A: float, I_z: float, I_y: float, I_p: float, J: float):
        """
        Validate material properties.
        
        :param E: Young's modulus.
        :param nu: Poisson's ratio.
        :param A: Cross-sectional area.
        :param I_z: Moment of inertia about the z-axis.
        :param I_y: Moment of inertia about the y-axis.
        :param I_p: Polar moment of inertia.
        :param J: Torsional constant.
        """
        if E <= 0:
            raise ValueError("Young's modulus (E) must be positive.")
        if not (-1 <= nu <= 0.5):
            raise ValueError("Poisson's ratio (nu) must be between -1 and 0.5.")
        if A <= 0:
            raise ValueError("Cross-sectional area (A) must be positive.")
        if I_z <= 0:
            raise ValueError("Moment of inertia about the z-axis (I_z) must be positive.")
        if I_y <= 0:
            raise ValueError("Moment of inertia about the y-axis (I_y) must be positive.")
        if I_p <= 0:
            raise ValueError("Polar moment of inertia (I_p) must be positive.")
        if J <= 0:
            raise ValueError("Torsional constant (J) must be positive.")

    def assign_subdomain(self, subdomain_id: int, element_ids: List[int]):
        """
        Assign elements to a subdomain.
        
        :param subdomain_id: ID of the subdomain.
        :param element_ids: List of element IDs to assign to the subdomain.
        """
        if subdomain_id not in self.materials:
            raise ValueError(f"Subdomain {subdomain_id} does not exist.")
        self.subdomains[subdomain_id] = element_ids


class BoundaryConditions:
    def __init__(self, supports: Dict[int, Tuple[float, float, float, float, float, float]]):
        """
        Initialize boundary conditions with a dictionary of supports.
        
        :param supports: Dictionary where keys are node IDs and values are tuples of support conditions.
                        Example: {node_id: (u_x, u_y, u_z, theta_x, theta_y, theta_z)}
        """
        self.loads = {}
        self.supports = supports
        self._validate_supports()

    def _validate_supports(self):
        """
        Validate the supports dictionary.
        """
        for node_id, support_conditions in self.supports.items():
            if not isinstance(node_id, int) or node_id < 0:
                raise ValueError(f"Node ID {node_id} must be a non-negative integer.")
            if len(support_conditions) != 6:
                raise ValueError(f"Support conditions for node {node_id} must be a tuple of length 6.")
            for condition in support_conditions:
                if condition is not None and not isinstance(condition, (float, int)):
                    raise ValueError(f"Support condition {condition} for node {node_id} must be a float, int, or None.")

    def _validate_loads(self):
        """
        Validate the loads dictionary.
        """
        for node_id, load_values in self.loads.items():
            if not isinstance(node_id, int) or node_id < 0:
                raise ValueError(f"Node ID {node_id} must be a non-negative integer.")
            if len(load_values) != 6:
                raise ValueError(f"Load values for node {node_id} must be a tuple of length 6.")
            for value in load_values:
                if not isinstance(value, (float, int)):
                    raise ValueError(f"Load value {value} for node {node_id} must be a float or int.")

    def add_load(self, loads: Dict[int, Tuple[float, float, float, float, float, float]]):
        """
        Add a nodal load.
        
        :param loads: Dictionary where keys are node IDs and values are tuples of load values.
                     Example: {node_id: (Fx, Fy, Fz, Mx, My, Mz)}
        """
        self.loads = loads
        self._validate_loads()

    def add_fixed_support(self, node_id: int):
        """Add a fixed support at the specified node."""
        self.supports[node_id] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._validate_supports()

    def add_pinned_support(self, node_id: int):
        """Add a pinned support at the specified node."""
        self.supports[node_id] = (0, 0, 0, None, None, None)
        self._validate_supports()

        
class Solver:
    def __init__(self, mesh: Mesh, material_params: MaterialParams, boundary_conditions: BoundaryConditions):
        """
        Initialize the solver with mesh, material properties, and boundary conditions.
        
        :param mesh: Mesh object containing nodes and elements.
        :param material_params: MaterialParams object containing material properties and subdomains.
        :param boundary_conditions: BoundaryConditions object containing supports and loads.
        """
        self.mesh = mesh
        self.material_params = material_params
        self.boundary_conditions = boundary_conditions

    def assemble_global_stiffness_matrix(self) -> np.ndarray:
        """Assemble the global stiffness matrix."""
        n_nodes = len(self.mesh.nodes)
        n_dofs = n_nodes * 6  # 6 DOFs per node
        K_global = np.zeros((n_dofs, n_dofs))

        for element_id, (node1, node2) in enumerate(self.mesh.elements):
            subdomain_id = next((k for k, v in self.material_params.subdomains.items() if element_id in v), None)
            if subdomain_id is None:
                raise ValueError(f"Element {element_id} is not assigned to any subdomain.")
            props = self.material_params.materials[subdomain_id]

            L = np.linalg.norm(self.mesh.nodes[node2] - self.mesh.nodes[node1])
            k_local = mut.local_elastic_stiffness_matrix_3D_beam(props['E'], props['nu'], props['A'], L, props['I_y'], props['I_z'], props['J'])

            node1_loc = self.mesh.nodes[node1]
            node2_loc = self.mesh.nodes[node2]
            gamma = mut.rotation_matrix_3D(node1_loc[0], node1_loc[1], node1_loc[2], node2_loc[0], node2_loc[1], node2_loc[2], props['local_z_axis'])
            T = mut.transformation_matrix_3D(gamma)

            k_global = T.T @ k_local @ T

            dofs = np.array([node1*6, node1*6+1, node1*6+2, node1*6+3, node1*6+4, node1*6+5,
                             node2*6, node2*6+1, node2*6+2, node2*6+3, node2*6+4, node2*6+5])
            for i in range(12):
                for j in range(12):
                    K_global[dofs[i], dofs[j]] += k_global[i, j]

        return K_global

    def solve_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the system for unknown displacements and reaction forces."""
        K_global = self.assemble_global_stiffness_matrix()
        n_dofs = K_global.shape[0]
        displacements = np.zeros(n_dofs)
        reactions = np.zeros(n_dofs)

        known_dofs = []
        unknown_dofs = []

        for node_id, (u_x, u_y, u_z, theta_x, theta_y, theta_z) in self.boundary_conditions.supports.items():
            dofs = [node_id * 6 + i for i in range(6)]
            if u_x is not None:
                known_dofs.append(dofs[0])
                displacements[dofs[0]] = u_x
            if u_y is not None:
                known_dofs.append(dofs[1])
                displacements[dofs[1]] = u_y
            if u_z is not None:
                known_dofs.append(dofs[2])
                displacements[dofs[2]] = u_z
            if theta_x is not None:
                known_dofs.append(dofs[3])
                displacements[dofs[3]] = theta_x
            if theta_y is not None:
                known_dofs.append(dofs[4])
                displacements[dofs[4]] = theta_y
            if theta_z is not None:
                known_dofs.append(dofs[5])
                displacements[dofs[5]] = theta_z

        unknown_dofs = [i for i in range(n_dofs) if i not in known_dofs]

        K_uu = K_global[np.ix_(unknown_dofs, unknown_dofs)]
        K_uk = K_global[np.ix_(unknown_dofs, known_dofs)]
        K_ku = K_global[np.ix_(known_dofs, unknown_dofs)]
        K_kk = K_global[np.ix_(known_dofs, known_dofs)]

        F = np.zeros(n_dofs)
        for node_id, (Fx, Fy, Fz, Mx, My, Mz) in self.boundary_conditions.loads.items():
            dofs = [node_id * 6 + i for i in range(6)]
            F[dofs[0]] = Fx
            F[dofs[1]] = Fy
            F[dofs[2]] = Fz
            F[dofs[3]] = Mx
            F[dofs[4]] = My
            F[dofs[5]] = Mz

        F_u = F[unknown_dofs]
        F_k = F[known_dofs]

        displacements[unknown_dofs] = np.linalg.solve(K_uu, F_u)
        reactions[known_dofs] = K_ku @ displacements[unknown_dofs] - F_k

        n_nodes = int(n_dofs / 6)
        displacements = displacements.reshape(n_nodes, 6)
        reactions = reactions.reshape(n_nodes, 6)

        return displacements, reactions

    def generate_mesh_and_solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh and solve the system."""
        displacements, reactions = self.solve_system()

        for node_id, (disp, rxn) in enumerate(zip(displacements, reactions)):
            u, v, w, theta_x, theta_y, theta_z = np.around(disp, decimals=5)
            Fx, Fy, Fz, Mx, My, Mz = np.around(rxn, decimals=5)
            print(f'node {node_id} disp: [u:{u}, v:{v}, w:{w}, theta_x:{theta_x}, theta_y:{theta_y}, theta_z:{theta_z}]')
            print(f'node {node_id} rxn: [Fx:{Fx}, Fy:{Fy}, Fz:{Fz}, Mx:{Mx}, My:{My}, Mz:{Mz}]')
            print('------------------------------------------------------------------')

        return displacements, reactions