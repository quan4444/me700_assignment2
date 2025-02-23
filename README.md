# Assignment 2

[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/sandialabs/sibl#license)

[![codecov](https://codecov.io/gh/quan4444/me700_assignment2/graph/badge.svg?token=03HMSXAI2Z)](https://codecov.io/gh/quan4444/me700_assignment2)
[![tests](https://github.com/quan4444/me700_assignment2/actions/workflows/tests.yml/badge.svg)](https://github.com/quan4444/me700_assignment2/actions)

## Table of Contents

* [Direct Stiffness Method](#dsm)
* [Installation Instructions](#install)
* [Example - Single line](#sl)
* [Template to Run Code](#template)

## Direct Stiffness Method <a name="dsm"></a>

In structural engineering, the direct stiffness method, also known as the matrix stiffness method, is a structural analysis technique particularly suited for computer-automated analysis of complex structures including the statically indeterminate type [1]. It is a matrix method that makes use of the members' stiffness relations for computing member forces and displacements in structures. The direct stiffness method is the most common implementation of the finite element method (FEM). In applying the method, the system must be modeled as a set of simpler, idealized elements interconnected at the nodes. The material stiffness properties of these elements are then, through linear algebra, compiled into a single matrix equation which governs the behaviour of the entire idealized structure. The structure’s unknown displacements and forces can then be determined by solving this equation. The direct stiffness method forms the basis for most commercial and free source finite element software. 

[1] https://en.wikipedia.org/wiki/Direct_stiffness_method

## Installation Instructions <a name="install"></a>

To install this package, please begin by setting up a conda environment:
```bash
mamba create --name assignment2 python=3.9.12
```
Once the environment has been created, activate it:

```bash
mamba activate assignment2
```
Double check that python is version 3.9.12 in the environment:
```bash
python --version
```
Ensure that pip is using the most up to date version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```
Create an editable install of the direct stiffness method code (note: you must be in the same directory as the .toml file):
```bash
python -m pip install -e .
```

OPTIONAL for debugging: If any package is missing, you can simply run the installation command for that package. For example, if pytest is missing, please run:
```bash
mamba install pytest
```

Test that the code is working with pytest:
```bash
pytest -v --cov=a2 --cov-report term-missing
```

If you are using VSCode to run this code, don't forget to set VSCode virtual environment to assignment2 environment.

To run the tutorials notebook, ``run_direct_stiffness_method.ipynb`` (located inside the ``tutorials`` folder), you can install notebook as follow:
```bash
mamba install jupyter notebook -y
```
To run the notebook in VSCode, simply double click on the ``run_direct_stiffness_method.ipynb`` file. If you're on a terminal on a desktop, you can run the notebook with:
```bash
jupyter notebook run_direct_stiffness_method.ipynb
```
_Note: Depending on your IDE, you might be prompted to install some extra dependencies. In VSCode, this should be a simple button click._


## Example problem - Single line <a name="sl"></a>

We have a single beam with 2 connected nodes, one at $[0,0,0]$, and one at $[1,0,0]$. The left node is fixed, and the right node experiences an external force $F_y=-1000$. The beam has the following properties $E=210e9$,$\nu=0.3$,$A=0.01$,$I_z=1e-5$,$I_y=1e-5$,$I_p=1e-5$,$J=1.0$, and local z-axis $[0,0,1]$. Solve for the missing displacements and reaction forces at the fixed node.


```python
import numpy as np
from a2 import direct_stiffness_method as dsm

############### USER INPUTS ################

nodes = np.array([[0, 0, 0], [1, 0, 0]])
elements = np.array([[0, 1]])
subdomain_dict = {1:[210e9,0.3,0.01,1e-5,1e-5,1e-5,1e-5,[0,0,1]]}
subdomain_elements = {1:[0]}
list_fixed_nodes_id = [0]
list_pinned_nodes_id = []
load_dict = {1:[0,-1000,0,0,0,0]}

############################################

disps,rxns = dsm.generate_mesh_and_solve(
        nodes,elements,subdomain_dict,subdomain_elements,list_fixed_nodes_id,list_pinned_nodes_id,load_dict
    )
```


```python
for node_id,(disp,rxn) in enumerate(zip(disps,rxns)):
    print(f'node {node_id} has displacements {disp} and reaction forces {rxn}')
    print('------------------------------------------------------------------')
```

    node 0 has displacements [0. 0. 0. 0. 0. 0.] and reaction forces [    0.  1000.     0.     0.     0. -1000.]
    ------------------------------------------------------------------
    node 1 has displacements [ 0.         -0.00015873  0.          0.          0.          0.0002381 ] and reaction forces [0. 0. 0. 0. 0. 0.]
    ------------------------------------------------------------------


## Template to run code - Put your mesh and problem here! <a name="template"></a>

In the ``tutorials`` folder, open ``run_direct_stiffness_method,ipynb``. Here we provide a blank template to run the code. Please provide the inputs as follow:

**INPUTS**:

- _nodes_: a n by 3 array containing the 3D coordinates of the mesh, where n is the number of nodes.

- _elements_: a m by 2 array containing the connectivities of the mesh, where m is the number of elements.

- _subdomain_dict_: a dictionary containing the properties of the material, where the key stores the ID for the subdomain, and the value a list of properties.
        e.g., subdomain 1 with list of properties E, nu, A, I_z, I_y, I_p, J, local_z_axis - 
        ``{1:[E,nu,A,I_z,I_y,I_p,J,local_z_axis]}``

- _subdomain_elements_: a dictionary containing the elements subdomain assignments.
        e.g., subdomain 1 has elements 0,1 - ``{1:[0,1]}``

- _list_fixed_nodes_id_: a list containing the node IDs for fixed nodes.
        e.g., ``[0]``

- _list_pinned_nodes_id_: a list containing the node IDs for pinned nodes.

- _load_dict_: a dictionary where the key is the node ID of the load applied, and the value the list representing the load applied.
        e.g., ``{node_id:[F_x,F_y,F_z,M_x,M_y,M_z]}``

**OUTPUTS**:

- _disps_: n by 6 array containing the global nodal displacements ``[u_x,u_y,u_z,theta_x,theta_y,theta_z]``

- _rxns_: n by 6 array containing the global rxn forces ``[F_x,F_y,F_z,M_x,M_y,M_z]``


```python
import numpy as np
from a2 import direct_stiffness_method as dsm

############### USER INPUTS ################

nodes = np.array([[0, 0, 0], [1, 0, 0]])
elements = np.array([[0, 1]])
subdomain_dict = {1:[210e9,0.3,0.01,1e-5,1e-5,1e-5,1e-5,[0,0,1]]}
subdomain_elements = {1:[0]}
list_fixed_nodes_id = [0]
list_pinned_nodes_id = []
load_dict = {1:[0,-1000,0,0,0,0]}

############################################

disps,rxns = dsm.generate_mesh_and_solve(
        nodes,elements,subdomain_dict,subdomain_elements,list_fixed_nodes_id,list_pinned_nodes_id,load_dict
    )
```


```python
for node_id,(disp,rxn) in enumerate(zip(disps,rxns)):
    print(f'node {node_id} has displacements {disp} and reaction forces {rxn}')
    print('------------------------------------------------------------------')
```
