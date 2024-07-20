#Script to quickly generate data for egno inflation model using inflatox
import sympy
import numpy as np
import inflatox

print("[EGNO supergravity inflation script]")
model = "egno"

################################################################################
#                                 model set-up                                 #
################################################################################

alpha, m, p, c, a = sympy.symbols('alpha m p c a')
r, θ = sympy.symbols('r θ')
fields = [r, θ]

Phi, Phi_Bar, S, S_Bar = sympy.symbols('Phi Phi_B S S_B')

K = (-3*alpha*sympy.ln(
  Phi + Phi_Bar - c*((Phi + Phi_Bar - 1))**4
) + (S*S_Bar)/(Phi + Phi_Bar)**3).nsimplify()

superfields = [Phi, S]
superfields_conjugate = [Phi_Bar, S_Bar]
metric = [[sympy.diff(sympy.diff(K, superfields[b]), superfields_conjugate[a]) for a in range(0,2)] for b in range(0,2)]
metric = [[g.subs({Phi: r+1j*θ, Phi_Bar:r-1j*θ}).nsimplify().simplify() for g in gb] for gb in metric]
metric = [[g.subs({S: 0, S_Bar: 0}).simplify() for g in gb] for gb in metric]
real_metric = [
  [metric[0][0], 0],
  [0, metric[0][0]]
]

print(f"metric tensor: {real_metric}")

potential = (
  (6*m**2*r**3*((a-r)**2+θ**2)) / (a**2*(2*r-c*(1-2*r)**4)**(3*alpha))
).nsimplify()

print(f"potential: {potential}")

################################################################################
#                                 compile model                                #
################################################################################

hesse = inflatox.SymbolicCalculation.new_from_list(
  fields,
  real_metric,
  potential,
  simplify_for='length',
  simplification_depth=1,
  silent=True
).execute([[0,1]])

out = inflatox.Compiler(hesse, silent=False).compile()
out.print_sym_lookup_table()

################################################################################
#                                   parameters                                 #
################################################################################

from inflatox.consistency_conditions import GeneralisedAL
anguelova = GeneralisedAL(out)

alpha = 1.0
a = 0.5
c = 1000.0
p = 3.055
m = 1e-3
args = np.array([m, a, c, alpha])

r_start, r_stop = 0.45, 0.55
θ_start, θ_stop = 0.0, np.pi
N_r, N_θ = 5000, 1000

extent = (0.46, 0.50, θ_start, θ_stop)

################################################################################
#                             run and save numerics                            #
################################################################################

#Calculate potential
potential = anguelova.calc_V_array(
  args,
  [r_start, r_stop],
  [θ_start, θ_stop],
  [N_r, N_θ]
)
np.save(f"./out/{model}_potential.npy", potential)
del potential

#run analysis
consistency, epsilon_V, epsilon_H, eta_H, delta, omega = anguelova.complete_analysis(
  args,
  *extent,
  *[N_r, N_θ]
)
np.save(f"./out/{model}.npy", consistency)
np.save(f"./out/{model}_epsilon_V.npy", epsilon_V)
np.save(f"./out/{model}_epsilon_H.npy", epsilon_H)
np.save(f"./out/{model}_eta_H.npy", eta_H)
np.save(f"./out/{model}_delta.npy", delta)
np.save(f"./out/{model}_omega.npy", omega)

#run analysis on trajectory
tr = np.load('./trajectories/egno_r.npy')
ttheta = np.load('./trajectories/egno_theta.npy')
trajectory = np.column_stack((tr, ttheta))

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
  *[N_r, N_θ]
)
np.save(f"./out/{model}_old.npy", consistency_old)