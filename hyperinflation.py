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

M, mh = sympy.symbols('M m_h')
metric = [
  [(1 + 2*chi**2/M**2), 0],
  [0, 1]
]

V0 = sympy.symbols('V0')
potential = 

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
  model_name="hyperinflation"
).execute([[0,1]])

out = inflatox.Compiler(hesse).compile()
out.print_sym_lookup_table()

################################################################################
#                                   parameters                                 #
################################################################################

from inflatox.consistency_conditions import AnguelovaLazaroiuCondition
anguelova = AnguelovaLazaroiuCondition(out)

V0 = -1.17e-8
N = 1000.0
gs = 0.01
ls = 501.961
u = 50*ls
q = 1.0
p = 5.0
a0 = 0.001
a1 = 0.0005
b1 = 0.001

parameters = np.array([V0, a0, p, q, u, ls, a1, b1, gs, N])

r_start, r_stop = 0.0, 36.0
θ_start, θ_stop = 0.0, 4*np.pi
extent = (r_start, r_stop, θ_start, θ_stop)
N = 1200

################################################################################
#                             run and save numerics                            #
################################################################################

#Calculate potential
potential = anguelova.calc_V_array(
  parameters,
  [r_start, r_stop],
  [θ_start, θ_stop],
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
trajectory = np.loadtxt('./trajectories/d5_trajectory.dat')

consistency, epsilon_V, epsilon_H, eta_H, delta, omega = anguelova.complete_analysis_on_trajectory(
  parameters,
  trajectory
)
np.save(f"./out/{model}_ot.npy", consistency)
np.save(f"./out/{model}_ot_epsilon_V.npy", epsilon_V)
np.save(f"./out/{model}_ot_epsilon_H.npy", epsilon_H)
np.save(f"./out/{model}_ot_eta_H.npy", eta_H)
np.save(f"./out/{model}_ot_delta.npy", delta)
np.save(f"./out/{model}_ot_omega.npy", omega)

#run Anguelova's original condition
consistency_old = anguelova.consistency_only_old(
  parameters,
  *extent,
  *[N, N]
)
np.save(f"./out/{model}_old.npy", consistency_old)