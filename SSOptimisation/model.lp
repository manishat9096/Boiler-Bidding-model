\ Model Steady_state_1MTU_problem
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  - 28.97125 Ecleared_down + [ 57.9425 u_Act * Ecleared_down ] / 2 
Subject To
 c1: - 0.25 C_up >= -31
 c2: 0.25 C_down <= 11
 c3: C_up >= 0
 c4: C_down >= 0
 c5: P_up + 14903.129997 u_Act = 15000
 c6: P_down - 15000 u_Act = 0
 c8: P_up + 1e+06 u_Act + 1e+06 ifCleared <= 2e+06
 c9: - P_down + 1e+06 u_Act - 1e+06 ifCleared >= -999942.0575
 c101: - 0.25 C_up + Ecleared_up <= 0
 c102: - 1e+06 ifCleared + Ecleared_up <= 0
 c103: - 0.25 C_up + 1e+06 ifCleared + Ecleared_up >= 1e+06
 c111: - 0.25 C_down + Ecleared_down <= 0
 c112: - 1e+06 ifCleared + Ecleared_down <= 0
 c113: - 0.25 C_down - 1e+06 ifCleared + Ecleared_down >= -1e+06
 System_state_after_bid: 0.5 Ecleared_down - Esys + [
   - 0.5 u_Act * Ecleared_up - 0.5 u_Act * Ecleared_down ] = -40
Bounds
 C_up <= 12
 C_down <= 12
 Ecleared_up <= 3
 Ecleared_down <= 3
Binaries
 u_Act ifCleared
Generals
 C_up C_down
End
