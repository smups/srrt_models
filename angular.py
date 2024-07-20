#Script to quickly generate data for angular inflation model using inflatox
import sympy
import numpy as np
import inflatox

print("[Angular inflation script]")
model = "angular"

################################################################################
#                                 model set-up                                 #
################################################################################

#setup the coordinates
p, x = sympy.symbols("phi chi")
coords = [p, x]
d = len(coords)

#setup the potential
mp, mx, a = sympy.symbols("m_phi m_chi alpha")
V = a/2 * ( (mp*p)**2 + (mx*x)**2 ).nsimplify()

#setup the metric
metric_diagonal = 6*a / (1 - p**2 - x**2)**2
metric = [[0 for _ in range(2)] for _ in range(2)]
metric[0][0] = metric_diagonal
metric[1][1] = metric_diagonal

print(f"metric tensor: {metric}")

################################################################################
#                                 compile model                                #
################################################################################

hesse = inflatox.SymbolicCalculation.new_from_list(
  coords,
  metric,
  V,
  model_name="angular inflation",
  assertions=False,
  simplification_depth=1,
  silent=True
).execute([[0,1]])

out = inflatox.Compiler(hesse, cleanup=False).compile()
out.print_sym_lookup_table()

################################################################################
#                                   parameters                                 #
################################################################################

from inflatox.consistency_conditions import GeneralisedAL
anguelova = GeneralisedAL(out)

a = 1/600
m_phi = 2e-5
m_chi = m_phi * np.sqrt(9)
args = np.array([a, m_chi, m_phi])

extent = (-1.05, 1.05, -1.05, 1.05)
phi_start, phi_stop = -15.0, 15.0
chi_start, chi_stop = -5.0, 5.0
N = 5000

################################################################################
#                             run and save numerics                            #
################################################################################

#Calculate potential
potential = anguelova.calc_V_array(
  args,
  [phi_start, chi_start],
  [phi_stop, chi_stop],
  [N, N]
)
np.save("./out/angular_potential.npy", potential)
del potential

#run analysis
consistency, epsilon_V, epsilon_H, eta_H, delta, omega = anguelova.complete_analysis(
  args,
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
tx = np.load('./trajectories/angular_phix.npy')
ty = np.load('./trajectories/angular_phiy.npy')
trajectory = np.column_stack((tx, ty))

consistency, epsilon_V, epsilon_H, eta_H, delta, omega = anguelova.complete_analysis_ot(
  args,
  trajectory
)
np.save(f"./out/{model}_ot.npy", consistency)
np.save(f"./out/{model}_ot_epsilon_V.npy", epsilon_V)
np.save(f"./out/{model}_ot_epsilon_H.npy", epsilon_H)
np.save(f"./out/{model}_ot_eta_H.npy", eta_H)
np.save(f"./out/{model}_ot_delta.npy", delta)
np.save(f"./out/{model}_ot_omega.npy", omega)

#run Anguelova's original condition
consistency_old = anguelova.consistency_rapidturn(
  args,
  *extent,
  *[N, N]
)
np.save(f"./out/{model}_old.npy", consistency_old)