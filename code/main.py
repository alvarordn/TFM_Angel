# Importing required libraries
import numpy as np
import lib


Sbase = 1e6
Ubase = 20e3
Zbase = (Ubase**2)/Sbase

# Nodes
Nodes = [{'id': 0, 'slack': True },
         {'id': 1, 'slack': False},
         {'id': 2, 'slack': False},
         {'id': 3, 'slack': False},]

# Lines
Lines = [{'id': 0,  'From': 0,  'To': 1,  'R': 0.161*4/Zbase, 'X': 0.109*4/Zbase}, 
         {'id': 1,  'From': 1,  'To': 2,  'R': 0.161*2/Zbase, 'X': 0.109*2/Zbase}, 
         {'id': 2,  'From': 2,  'To': 3,  'R': 0.161*5/Zbase, 'X': 0.109*5/Zbase}]

# Prosumers
Pros = [{'id': 0, 'Node': 1, 'P': -2e6/Sbase, 'Q': -1.5e6/Sbase},
        {'id': 1, 'Node': 2, 'P': -1.6e6/Sbase, 'Q': -1.2e6/Sbase},
        {'id': 2, 'Node': 3, 'P': -6.4e6/Sbase, 'Q': -2.4e6/Sbase}]

# Constructing network and solving power flow
net = lib.grid(Nodes, Lines, Pros)

sol = net.solve_pf()

U = net.compute_voltages()
I_lines = net.compute_line_currents()
I_nodes = net.compute_node_currents()
check = net.check() 

if all(check):
    print('The solution is consistent')
    
    
    
    
    
    
    
    
    
    

