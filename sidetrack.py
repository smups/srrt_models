#Script to quickly generate data for d5-brane inflation model using inflatox
import sympy
from sympy.simplify.radsimp import collect_sqrt
import numpy as np
import inflatox

print("[Side-tracked inflation script]")
model = "sidetrack"

################################################################################
#                                 model set-up                                 #
################################################################################

phi, chi = sympy.symbols('φ χ')
fields = [phi, chi]

M, m, f = sympy.symbols('M m f')
metric = [
  [(1 + 2 *(chi/M)**2).nsimplify(), sympy.sqrt(2)*chi/M],
  [sympy.sqrt(2)*chi/M, 1]
]

potential = (1/2 * (m * chi)**2 + 1 + sympy.cos(phi/f)).nsimplify()

print(f"metric tensor: {metric}")
print(f"potential: {potential}")

################################################################################
#                                 compile model                                #
################################################################################

hesse = inflatox.SymbolicCalculation.new_from_list(
  fields,
  metric,
  potential,
  assertions=False,
  simplification_depth=1,
  silent=True,
  model_name="side-tracked inflation (1804.11279)"
).execute([[0,1]])

out = inflatox.Compiler(hesse).compile()
out.print_sym_lookup_table()

################################################################################
#                                   parameters                                 #
################################################################################

from inflatox.consistency_conditions import AnguelovaLazaroiuCondition
anguelova = AnguelovaLazaroiuCondition(out)

M = 1e-3
m = 10
f = 10
parameters = np.array([m, f, M])

phi_start, phi_stop = -0.5*f, 3.5*f
chi_start, chi_stop = -3*M, 3*M 
extent = (phi_start, phi_stop, chi_start, chi_stop)
N = 1000

################################################################################
#                             run and save numerics                            #
################################################################################

#Calculate potential
potential = anguelova.calc_V_array(
  parameters,
  [phi_start, phi_stop],
  [chi_start, chi_stop],
  [N, N]
)
np.save(f"./out/{model}_potential.npy", potential)
del potential

#run analysis
consistency, epsilon_V, epsilon_H, eta_H, delta, omega = anguelova.complete_analysis(
  parameters,
  *extent,
  *[N, N]
)
np.save(f"./out/{model}.npy", consistency)
np.save(f"./out/{model}_epsilon_V.npy", epsilon_V)
np.save(f"./out/{model}_epsilon_H.npy", epsilon_H)
np.save(f"./out/{model}_eta_H.npy", eta_H)
np.save(f"./out/{model}_delta.npy", delta)
np.save(f"./out/{model}_omega.npy", omega)

#run analysis on trajectory
# trajectory = np.loadtxt('./trajectories/d5_trajectory.dat')

# consistency, epsilon_V, epsilon_H, eta_H, delta, omega = anguelova.complete_analysis_on_trajectory(
#   parameters,
#   trajectory
# )
# np.save(f"./out/{model}_ot.npy", consistency)
# np.save(f"./out/{model}_ot_epsilon_V.npy", epsilon_V)
# np.save(f"./out/{model}_ot_epsilon_H.npy", epsilon_H)
# np.save(f"./out/{model}_ot_eta_H.npy", eta_H)
# np.save(f"./out/{model}_ot_delta.npy", delta)
# np.save(f"./out/{model}_ot_omega.npy", omega)

#run Anguelova's original condition
consistency_rapidturn = anguelova.consistency_rapidturn(
  parameters,
  *extent,
  *[N, N]
)
np.save(f"./out/{model}_old.npy", consistency_rapidturn)