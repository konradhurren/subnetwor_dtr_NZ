import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple



### List of variables and where they are calculated

### from the raw data csv file
# W_it: Available capacity at node i, time t. This is the rating of the node.
# F_it: Flow at node i, time t (load + supply).
# L_it: Load at node i, time t.
# S_it: Supply at node i, time t.

### calculated using these functions
# H_it: Headroom at node i, time t.
# U_it: Signed utilization at node i, time t.
# DDM_it: Delivery discount multiplier at node i, time t.
# b_ijt: Basis at node i, time t.
# Vd_ijt: Delivery-adjusted value at node i, time t.

## calculated using the optimal generator schedule
# J_s: 1 if the generator is buying/importing, -1 if the generator is selling/exporting.
# J_s is calculated in the optimal generator schedule.  

## set by the user
# phi: Scaling parameter (phi > 0).
# alpha_ldf: Exponent parameter for the delivery factor (alpha > 0).

def calculate_headroom(W_it, F_it):
    """
    (1a) Calculates the headroom at a specific node and time.
    Headroom is the difference between the available capacity and the current flow.

    Args:
        W_it (float): Available capacity at node i, time t. This is the rating of the node.
        F_it (float): Flow at node i, time t (load + supply).

    Returns:
        float: The calculated headroom (H_it).
    """
    return W_it - F_it

def calculate_flow(L_it, S_it):
    """
    (2a) Calculates the total flow at a node, which is the sum of load and supply.

    Args:
        L_it (float): Load at node i, time t.
        S_it (float): Supply at node i, time t.

    Returns:
        float: The total flow (F_it).
    """
    return L_it - S_it



def calculate_signed_utilization(F_it, W_it, J_s) :
    """
    (3a) Calculates the signed utilization at a node and time.
    Signed utilization is the flow at a node and time divided by the available capacity at the node and time.
    Args:
        F_it (float): Flow at node i, time t.
        W_it (float): Available capacity at node i, time t.
        J_s (float): 1 if the generator is buying/importing, -1 if the generator is selling/exporting.
        J_s is calculated in the optimal generator schedule.
    Returns:
        float: The signed utilization (U_it).
    """
    if W_it == 0:
        return float('inf') if F_it != 0 else 0.0
    if J_s == 1:
        return F_it / W_it
    else:
        return -F_it / W_it
    
      

def calculate_global_headroom(headroom_values):
    """
    (4a) Calculates the global headroom as the sum of headrooms across all nodes.

    Args:
        headroom_values (list of float): A list containing the headroom (H_it)
                                         for each node in the system.

    Returns:
        float: The total global headroom (H_t).
    """
    return sum(headroom_values)

def calculate_congestion(H_it):
    """
    (6a) Calculates the congestion at a node as the inverse of its headroom.
    A small headroom implies high congestion.

    Args:
        H_it (float): Headroom at node i, time t.

    Returns:
        float: The congestion value (con_it). Returns float('inf') if headroom is 0.
    """
    if H_it == 0:
        return float('inf') # Represents infinite congestion
    return 1 / H_it


def calculate_delivery_discount_multiplier(U_it, phi, alpha_ldf):
    """
    (8a) Calculates the Delivery Discount Multiplier (DDM).

    Args:
        U_it (float): Signed utilization at node i, time t.
        phi (float): Scaling parameter (phi > 0).
        alpha_ldf (float): Exponent parameter for the delivery factor (alpha > 0).
    Returns:
        float: The Delivery Discount Multiplier (DDM_ij).
    """
    if U_it > 0:
        return 1 - (phi * max(0.0, U_it)) ** alpha_ldf
    else:
        return 1 + (phi * max(0.0, -U_it)) ** alpha_ldf

def calculate_basis(DDM_it, DDM_jt):
    """
    (9a) Calculates the basis, which is the difference in DDM between two nodes.

    Args:
        DDM_it (float): DDM at the source node i, time t.
        DDM_jt (float): DDM at the sink node j, time t.

    Returns:
        float: The basis (b_ij).
    """
    return DDM_it - DDM_jt

def calculate_delivery_adjusted_value(b_ijt, P_gxp_it):
    """
    (10a) Calculates the final delivery-adjusted value of a transaction.

    Args:
        b_ijt (float): The basis for the transaction between i and j at time t.
        P_gxp_it (float): The price of electricity at the closest grid exchange point.

    Returns:
        float: The delivery-adjusted value (Vd_ijt).
    """
    return b_ijt * P_gxp_it

def phys_check(L, S):
    """
    Checks if the sum of all L_it equals the sum of all S_it at each time t.

    Args:
        L (list of list of float): Loads at each node and time, shape (num_nodes, num_times).
        S (list of list of float): Supplies at each node and time, shape (num_nodes, num_times).

    Returns:
        list of bool: For each t, True if sum of L[:, t] == sum of S[:, t], else False.
    """
    if isinstance(L, float) and isinstance(S, float):
        return [True]  # or your preferred logic for floats
    if len(L) == 0 or len(S) == 0 or len(L[0]) != len(S[0]):
        raise ValueError("Input lists must be non-empty and have the same shape.")
    num_times = len(L[0])
    results = []
    for t in range(num_times):
        total_load = sum(L[i][t] for i in range(len(L)))
        total_supply = sum(S[i][t] for i in range(len(S)))
        results.append(total_load == total_supply)
    return results

