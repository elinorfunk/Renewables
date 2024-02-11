# Assignment 1 

import pandas as pd
import math
import sys
sys.path.append("/Users/elino/anaconda3/lib/python3.11/site-packages")
import gurobipy as gb

# Base power
SB = 100

# Sets
NODES = range(1,25) # 24 nodes
CONVENTIONAL = range(1,13) # 12 conventional generating units
WIND = range(1,5) # 4 wind farms

# Input data from excel file
excel_file = '/Users/elino/Documents/RenewablesIEM/Q1inputData.xlsx'

# Conventional: capacity and cost
df_conv = pd.read_excel(excel_file, sheet_name='conv')

# Wind: available production for single hour and cost (zero)
df_w = pd.read_excel(excel_file, sheet_name='wind')

# Hourly demand - single hour will be selected
df_demand_h = pd.read_excel(excel_file, sheet_name='hourlyDemand')

# Load percentage in each node
df_load = pd.read_excel(excel_file, sheet_name='loadShare')

# Transmission: reactance and transfer capacity
df_trans = pd.read_excel(excel_file, sheet_name='transmission')

# Custom input: select single hour from demand data
DEMAND = df_demand_h.loc[0,'Demand'] # corresponds to Hour = 1

# Initialize zero matrices
X = pd.DataFrame(0,index=NODES,columns=NODES)
Tcap = pd.DataFrame(0,index=NODES,columns=NODES)
PmaxC = pd.DataFrame(0,index=CONVENTIONAL,columns=NODES)
PmaxW = pd.DataFrame(0,index=WIND,columns=NODES)
CostC = pd.DataFrame(0,index=CONVENTIONAL,columns=NODES)
CostW = pd.DataFrame(0,index=WIND,columns=NODES)

for i in range(0,len(df_trans)):
    # Define reactance matrix
    X.loc[df_trans.loc[i,'From'],df_trans.loc[i,'To']] = df_trans.loc[i,'Xpu']
    X.loc[df_trans.loc[i,'To'],df_trans.loc[i,'From']] = df_trans.loc[i,'Xpu']
    # Define transfer capacity matrix
    Tcap.loc[df_trans.loc[i,'From'],df_trans.loc[i,'To']] = df_trans.loc[i,'NTC']
    Tcap.loc[df_trans.loc[i,'To'],df_trans.loc[i,'From']] = df_trans.loc[i,'NTC']

for i in range(0,len(df_conv)):
    # Define maximum capacity of conventional units in a matrix
    # rowIndex = Unit Number, columnIndex = Node
    PmaxC.loc[df_conv.loc[i,'Unit'],df_conv.loc[i,'Node']] = df_conv.loc[i,'Pmax']
    # Define production costs of conventional units in a matrix
    # rowIndex = Unit Number, columnIndex = Node
    CostC.loc[df_conv.loc[i,'Unit'],df_conv.loc[i,'Node']] = df_conv.loc[i,'Ci']

for i in range(0,len(df_w)):
    # Define wind production of each farm in a matrix
    # rowIndex = Wind Farm , columnIndex = Node
    PmaxW.loc[df_w.loc[i,'Unit'],df_w.loc[i,'Node']] = df_w.loc[i,'pW']
    # Define production cost of each farm in a matrix
    # rowIndex = Wind Farm , columnIndex = Node
    CostW.loc[df_w.loc[i,'Unit'],df_w.loc[i,'Node']] = df_w.loc[i,'Ci']    

# Multiply percentage loads with total demand (convert pu to MW)
df_load['loadShare'] = df_load['loadShare']*DEMAND

# Allocate load share to all 24 nodes
pL = pd.DataFrame(0,index=NODES,columns=['Load'])
for i in range(0,len(df_load)):
    pL.loc[df_load.loc[i,'Node']] = df_load.loc[i,'loadShare']
    
    
# Initialize zero admittance matrix
B = pd.DataFrame(0,index=NODES,columns=NODES)

# Define Admittance matrix
# non-diagonal elements
for i in NODES:
    for j in NODES:
        if X.loc[i, j] != 0:
            B.loc[i, j] = -1 / X.loc[i, j]
  
# diagonal elements 
for i in NODES:
    B.loc[i,i] = sum(-(-1 / X.loc[i, j]) if X.loc[i, j] != 0 else 0 for j in NODES)


# Create optimization model
model = gb.Model("Single-time step OPF")

# Set time limit
model.Params.TimeLimit = 100

# Add variables
theta = {}
pG = {}
pW = {}
# Voltage angle in each node (1x24)
for n in NODES:  
    theta[n] = model.addVar(lb=-float('inf'), ub=float('inf'), name=f'Angle of Node {n}')
    
# Power generation of generating unit c in node n (12x24)  
for c in CONVENTIONAL:      
    for n in NODES:
        pG[c,n] = model.addVar(lb = -float('inf'), ub=float('inf'),
                               name=f'Power of conventional generator {c} in Node {n}')
        
# Power generation of wind farm w in node n (12x24)  
for w in WIND:       
    for n in NODES:
        pW[w,n] = model.addVar(lb = -float('inf'), ub=float('inf'),
                               name=f'Power of wind farm {w} in Node {n}')
    
    
# Define objective function
objective = gb.quicksum(CostC.loc[c, n] * pG[c, n] for c in CONVENTIONAL for n in NODES)
model.setObjective(objective, gb.GRB.MINIMIZE)


# Add constraints to the model

# Balance equation at each node
balance_constr = {n:model.addLConstr(gb.quicksum(pG[c,n] for c in CONVENTIONAL) 
                                        + gb.quicksum(pW[w,n] for w in WIND) 
                                        - gb.quicksum(pL.loc[n]),
                                        gb.GRB.EQUAL,
                                        gb.quicksum(B.loc[n,m]*theta[m]*SB for m in NODES),
                                        name=f'Balance equation at node {n}') for n in NODES}

# Max flow of transmission lines
maxFlowConstr = {(n,m):model.addLConstr( 
        -B.loc[n,m]*(theta[n]-theta[m])*SB,
        gb.GRB.GREATER_EQUAL,
        -Tcap.loc[n,m],
        name=f'Constraint on max flow between nodes {n} and {m}')
    for n in NODES for m in [node for node in NODES if node not in [n]]}

# Reverse maximum (negative) flow of transmission lines
revMaxFlowConstr = {(n,m):model.addLConstr( 
        B.loc[n,m]*(theta[n]-theta[m])*SB,
        gb.GRB.GREATER_EQUAL,
        -Tcap.loc[n,m],
        name='Constraint on reverse max flow between nodes {n} and {m}')
    for n in NODES for m in [node for node in NODES if node not in [n]]}

# Max production of generators
maxConvProductionConstr = {(c,n):model.addLConstr(
        -pG[c,n],
        gb.GRB.GREATER_EQUAL,
        -PmaxC.loc[c,n],name=f'Constraint on max production of generator {c}') 
    for c in CONVENTIONAL for n in NODES}

# Min production of generators
minConvProductionConstr = {(c,n):model.addLConstr(
        pG[c,n],
        gb.GRB.GREATER_EQUAL,
        0,name='Constraint on min production of generator {c}')
    for c in CONVENTIONAL for n in NODES}

# Max production of wind farms (based on available energy)
maxWindProductionConstr = {(w,n):model.addLConstr(
        -pW[w,n],
        gb.GRB.GREATER_EQUAL,
        -PmaxW.loc[w,n],name=f'Constraint on max production of wind farm {w}') 
    for w in WIND for n in NODES}

# Min production of wind farms
minWindProductionConstr = {(w,n):model.addLConstr(
        pW[w,n],
        gb.GRB.GREATER_EQUAL,
        0,name=f'Constraint on min production of wind farm {w}')
    for w in WIND for n in NODES}
              
# Node 1 is used as a reference for voltage angle              
voltageAngleRef = model.addLConstr(theta[1],"=",0,
                           name = 'Node 1 as voltage angle reference')

# Minimum Voltage Angle             
#minVoltageAngle = {n:model.addLConstr(theta[n],gb.GRB.GREATER_EQUAL,-math.pi,
 #                          name = f'Minimum voltage angle in node {n}')
  #                    for n in NODES}

# Maximum Voltage Angle             
#maxVoltageAngle = {n:model.addLConstr(-theta[n],gb.GRB.GREATER_EQUAL,-math.pi,
 #                          name = f'Maximum voltage angle in node {n}')
  #                    for n in NODES}


#Optimize the model
model.optimize()

# List all variables and constraints of Single-time step OPF problem in a vector
variables = model.getVars()
constraints = model.getConstrs()

# Results stored in Excel file
with pd.ExcelWriter('Q1_results.xlsx', engine='xlsxwriter') as writer:

    # Check if the optimization was successful and store results in separate sheets
    if model.status == GRB.OPTIMAL:
        
        # Store objective value in 'objValue' sheet
        optimal_objective = model.ObjVal
        obj_df = pd.DataFrame({"Variable": ["Objective Value"], "Value": [optimal_objective]})
        obj_df.to_excel(writer, sheet_name='objValue', index=False)
        
        # save values of variables of primal optimization problem at optimality
        optimal_variables = [variables[v].x for v in range(len(variables))]
        # optimal flows from node n to node m
        optimal_flows = {(m,n):B.loc[n,m]*(theta[n].x-theta[m].x)*SB for n in NODES for m in [node for node in NODES if node not in [n]]} 
        #save value of dual variables associated with each primal constraint at optimality
        optimal_sensitivities = [constraints[c].Pi for c in range(len(constraints))] 
        

        # Store voltage angles in 'angles' sheet
        angles_dict = {"Variable": [f"Angle of Node {n}" for n in NODES], 
                       "Node": [n for n in NODES],
                       "Value": [theta[n].x for n in NODES]}
        angles_df = pd.DataFrame(angles_dict)
        angles_df.to_excel(writer, sheet_name='angles', index=False)

        # Store conventional generation in 'pG' sheet
        power_dict = {"Variable": [f"Power of conventional generator {c} in Node {n}" for c in CONVENTIONAL for n in NODES  if (PmaxC.loc[c,n]>0)], 
                      "Value": [pG[c, n].x for c in CONVENTIONAL for n in NODES if PmaxC.loc[c,n]>0]}
        power_df = pd.DataFrame(power_dict)
        power_df.to_excel(writer, sheet_name='pG', index=False)
        
        # Store wind generation in 'pW' sheet
        wind_dict = {"Variable": [f"Power of wind farm {w} in Node {n}" for w in WIND for n in NODES  if (PmaxW.loc[w,n]>0)], 
                      "Value": [pW[w,n].x for w in WIND for n in NODES if PmaxW.loc[w,n]>0]}
        wind_df = pd.DataFrame(wind_dict)
        wind_df.to_excel(writer, sheet_name='pW', index=False)

        # Store dual variable values in 'duals' sheet
        duals_dict = {"Variable": [f"Dual variable for {constraints[c].constrName}" for c in range(len(constraints))], "Value": [optimal_sensitivities[c] for c in range(len(constraints))]}
        duals_df = pd.DataFrame(duals_dict)
        duals_df.to_excel(writer, sheet_name='duals', index=False)

        # Store flow values in 'flows' sheet
        flows_dict = {"Variable": [f"Flow from node {m} to {n}" 
                                   for n in NODES for m in [node for node in NODES if node not in [n]] if Tcap.loc[n,m]>0],
                      "Value": [optimal_flows[m, n] 
                                for n in NODES for m in [node for node in NODES if node not in [n]] if Tcap.loc[n,m]>0]}
        flows_df = pd.DataFrame(flows_dict)
        flows_df.to_excel(writer, sheet_name='flows', index=False)
        print("Optimization results have been saved to 'Q1_results.xlsx'.")

    else:
        print("Optimization was not successful.")
