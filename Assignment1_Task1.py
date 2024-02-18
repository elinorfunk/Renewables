import pandas as pd

import sys
sys.path.append("/Users/elino/anaconda3/lib/python3.11/site-packages")
import gurobipy as gb
import gurobipy as gb
from gurobipy import GRB
import matplotlib.pyplot as plt

CONVENTIONAL = range(1,13) # 12 conventional generating units
WIND = range(1,5) # 4 wind farms
DEMAND = range(1,18) # 17 loads

excel_file = '/Users/elino/Documents/RenewablesIEM/Task1_InputData.xlsx'

df_conv = pd.read_excel(excel_file, sheet_name='conv')
df_w = pd.read_excel(excel_file, sheet_name='wind')
df_load = pd.read_excel(excel_file, sheet_name='demand')

# Initialize zero matrices
PmaxC = pd.DataFrame(0, index=CONVENTIONAL, columns=['Value'])
PmaxW = pd.DataFrame(0,index=WIND,columns=['Value'])
CostC = pd.DataFrame(0,index=CONVENTIONAL,columns=['Value'])
CostW = pd.DataFrame(0,index=WIND,columns=['Value'])
pL = pd.DataFrame(0,index=DEMAND,columns=['Value'])
bid_price = pd.DataFrame(0,index=DEMAND,columns=['Value'])

for i in range(0,len(df_conv)):
    # Define maximum capacity of conventional units in a matrix
    # rowIndex = Unit Number, columnIndex = Node
    PmaxC.loc[df_conv.loc[i,'Unit']] = df_conv.loc[i,'Pmax']
    # Define production costs of conventional units in a matrix
    # rowIndex = Unit Number, columnIndex = Node
    CostC.loc[df_conv.loc[i,'Unit']] = df_conv.loc[i,'Ci']

for i in range(0,len(df_w)):
    # Define wind production of each farm in a matrix
    # rowIndex = Wind Farm , columnIndex = Node
    PmaxW.loc[df_w.loc[i,'Unit']] = df_w.loc[i,'pW']
    # Define production cost of each farm in a matrix
    # rowIndex = Wind Farm , columnIndex = Node
    CostW.loc[df_w.loc[i,'Unit']] = df_w.loc[i,'Ci']

for i in range(0,len(df_load)):
    pL.loc[df_load.loc[i,'Unit']] = df_load.loc[i,'Demand']
    bid_price.loc[df_load.loc[i,'Unit']] = df_load.loc[i,'Bid price']

# Create optimization model
model = gb.Model("Copper-plate single hour")

# Set time limit
model.Params.TimeLimit = 100

# Add variables
pG = {}
pW = {}

# Power generation of generating unit c in node n (12x24)  
for c in CONVENTIONAL:
    pG[c] = model.addVar(lb = -float('inf'), ub=float('inf'),
                               name=f'Power of conventional generator {c}')
    
# Power generation of wind farm w in node n (12x24)  
for w in WIND:
    pW[w] = model.addVar(lb = -float('inf'), ub=float('inf'),
                               name=f'Power of wind farm {w}')


# Define objective function
objective = gb.quicksum(CostC.loc[c].iloc[0] * pG[c] for c in CONVENTIONAL)
model.setObjective(objective, gb.GRB.MINIMIZE)

# Add constraints to the model

# Balance equation
balance_constr = {model.addLConstr(gb.quicksum(pG[c] for c in CONVENTIONAL) 
                                        + gb.quicksum(pW[w] for w in WIND) 
                                        - gb.quicksum(pL.loc[d] for d in DEMAND),
                                        gb.GRB.EQUAL,
                                        0,)
                  }


# Max production of generators
maxConvProductionConstr = {c:model.addLConstr(
        pG[c],
        gb.GRB.LESS_EQUAL,
        PmaxC.loc[c],
        name=f'Constraint on max production of generator {c}') 
    for c in CONVENTIONAL}

# Min production of generators
minConvProductionConstr = {c:model.addLConstr(
        -pG[c],
        gb.GRB.LESS_EQUAL,
        0,name='Constraint on min production of generator {c}')
    for c in CONVENTIONAL}

# Max production of wind farms (based on available energy)
maxWindProductionConstr = {w:model.addLConstr(
        pW[w],
        gb.GRB.LESS_EQUAL,
        PmaxW.loc[w],name=f'Constraint on max production of wind farm {w}') 
    for w in WIND}

# Min production of wind farms
minWindProductionConstr = {w:model.addLConstr(
        -pW[w],
        gb.GRB.LESS_EQUAL,
        0,name=f'Constraint on min production of wind farm {w}')
    for w in WIND}

#Optimize the model
model.optimize()

# List all variables and constraints of Single-time step OPF problem in a vector
variables = model.getVars()
constraints = model.getConstrs()

# Results stored in Excel file
with pd.ExcelWriter('Task1_results.xlsx', engine='xlsxwriter') as writer:

    # Check if the optimization was successful and store results in separate sheets
    if model.status == GRB.OPTIMAL:
        # Store objective value in 'objValue' sheet
        optimal_objective = model.ObjVal
        obj_df = pd.DataFrame({"Variable": ["Objective Value"], "Value": [optimal_objective]})
        obj_df.to_excel(writer, sheet_name='objValue', index=False)
        # save values of variables of primal optimization problem at optimality
        optimal_variables = [variables[v].x for v in range(len(variables))]
        #save value of dual variables associated with each primal constraint at optimality
        optimal_sensitivities = [constraints[c].Pi for c in range(len(constraints))]
        
        # Store conventional generation in 'pG' sheet
        power_dict = {"Variable": [f"Power of conventional generator {c}" for c in CONVENTIONAL], 
                      "Value": [pG[c].x for c in CONVENTIONAL]}
        power_df = pd.DataFrame(power_dict)
        power_df.to_excel(writer, sheet_name='pG', index=False)
        
        # Store wind generation in 'pW' sheet
        wind_dict = {"Variable": [f"Power of wind farm {w}" for w in WIND], 
                      "Value": [pW[w].x for w in WIND]}
        wind_df = pd.DataFrame(wind_dict)
        wind_df.to_excel(writer, sheet_name='pW', index=False)

        # Store dual variable values in 'duals' sheet
        duals_dict = {"Variable": [f"Dual variable for {constraints[c].constrName}" for c in range(len(constraints))], "Value": [optimal_sensitivities[c] for c in range(len(constraints))]}
        duals_df = pd.DataFrame(duals_dict)
        duals_df.to_excel(writer, sheet_name='duals', index=False)
        
        # Market-Clearing Price (Uniform Pricing Scheme)
        market_clearing_price = optimal_sensitivities[0]  # Dual variable associated with the balance equation constraint
        market_clearing_df = pd.DataFrame({"Variable": ["Market-Clearing Price"], "Value": [market_clearing_price]})
        market_clearing_df.to_excel(writer, sheet_name='Results', index=False)

        # Social Welfare
        social_welfare = gb.quicksum(pL.loc[d]*bid_price.loc[d] for d in DEMAND)-optimal_objective  # Objective value of the optimization problem
        welfare_df = pd.DataFrame({"Variable": ["Social Welfare"], "Value": [social_welfare]})
        welfare_df.to_excel(writer, sheet_name='Results', startrow=3, index=False)

        # Profit of Suppliers
        profit_conventional_units = {c: (market_clearing_price * pG[c].x - CostC.loc[c].iloc[0] * pG[c].x) for c in CONVENTIONAL}
        profit_wind_farms = {w: (market_clearing_price * pW[w].x - CostW.loc[w].iloc[0] * pW[w].x) for w in WIND}
        profit_df = pd.DataFrame({"Supplier": [f"Generator {c}" for c in CONVENTIONAL] + [f"Wind Farm {w}" for w in WIND],
                                  "Profit": [profit_conventional_units[c] for c in CONVENTIONAL] + [profit_wind_farms[w] for w in WIND]})
        profit_df.to_excel(writer, sheet_name='Results', startrow=6, index=False)

        # Utility of Demand
        # utility_demand = {d: market_clearing_price * pL.loc[d].iloc[0] for d in DEMAND}
        # utility_df = pd.DataFrame({"Demand": [f"Demand {d}" for d in DEMAND], "Utility": [utility_demand[d] for d in DEMAND]})
        # utility_df.to_excel(writer, sheet_name='Results', startrow=6+len(CONVENTIONAL)+len(WIND)+3, index=False)
        
    else:
        print("Optimization was not successful.")
        
# Extract demand and supply values
demand_quantity = [pL.loc[d].iloc[0] for d in DEMAND]
demand_price = [bid_price.loc[d].iloc[0] for d in DEMAND]

supply_quantity = [pG[c].x + pW[w].x for c in CONVENTIONAL for w in WIND]
supply_price = [market_clearing_price + (CostC.loc[c].iloc[0] if c in CONVENTIONAL else CostW.loc[w].iloc[0]) for c in CONVENTIONAL for w in WIND]

# Verify KKT conditions 
objective_gradients = [sum(CostC.loc[c].iloc[0] for c in CONVENTIONAL), 0]

# Manually cacluating the gradients of the constraints
gradient_balance_const = [[12, 4]]
gradients_conv_const = [[1, 0], [-1, 0]] * len(CONVENTIONAL)
gradients_wind_const = [[0, 1], [0, -1]] * len(WIND)
constraints_gradients = gradient_balance_const + gradients_conv_const + gradients_wind_const

# Calculating the dual variables times the gradients of the constraints 
gradients_times_dual = [[dual_var * constraints_gradients[i][0], dual_var * constraints_gradients[i][1]] for dual_var in optimal_sensitivities]

# Import numpy and convert the lists to arrays for easier addition 
import numpy as np 
obj_grad = np.array((objective_gradients))
dual_grad = np.array(gradients_times_dual)

# derivative_lagrangian should be equal to zero 
derivative_lagrangian = obj_grad + sum(dual_grad)

if derivative_lagrangian.all() == 0: 
    print("KKT-conditions are fulfilled")
else: 
    print("KKT-condtions are not fulfilled")

# Print the values for the duals times the gradients
# print(derivative_lagrangian)
# print(obj_grad)
# print(sum(dual_grad))
