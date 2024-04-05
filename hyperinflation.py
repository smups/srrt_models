#Script to quickly generate data for d5-brane inflation model using inflatox
import sympy
from sympy.simplify.radsimp import collect_sqrt
import numpy as np
import inflatox

print("[Hyperinflation inflation script]")
model = "hyper"

################################################################################
#                                 model set-up                                 #
################################################################################

phi, chi = sympy.symbols('φ χ')
fields = [phi, chi]

# M = ~10^-3

L, m = sympy.symbols('L m')
metric = [
  [sympy.cosh(chi/L)**2, 0],
  [0, 1]
]

potential = (1/2 * (m * phi)**2 + 1/2 * (m * chi)**2 * (phi / L)).nsimplify()

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
  model_name="hyperinflation (Christodoulidis 1903.03513)"
).execute([[0,1]])

out = inflatox.Compiler(hesse).compile()
out.print_sym_lookup_table()

################################################################################
#                                   parameters                                 #
################################################################################

from inflatox.consistency_conditions import AnguelovaLazaroiuCondition
anguelova = AnguelovaLazaroiuCondition(out)

L = 0.01
m = 1
parameters = np.array([m, L])

phi_start, phi_stop = 0, 80
chi_start, chi_stop = -.05, .05
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
# consistency_old = anguelova.consistency_only_old(
#   parameters,
#   *extent,
#   *[N, N]
# )
# np.save(f"./out/{model}_old.npy", consistency_old)