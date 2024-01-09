#Script to quickly generate data for egno inflation model using inflatox
import sympy
import numpy as np
import inflatox

print("[EGNO supergravity inflation script]")

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

out = inflatox.Compiler(hesse).compile()
out.print_sym_lookup_table()

################################################################################
#                                   paramters                                  #
################################################################################

from inflatox.consistency_conditions import AnguelovaLazaroiuCondition
anguelova = AnguelovaLazaroiuCondition(out)

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
# exact = anguelova.evaluate(args, *extent, N_x0=N_r, N_x1=N_θ, order='exact')
# np.save("./out/egno_exact.npy", exact)
# del exact
# leading = anguelova.evaluate(args, *extent, N_x0=N_r, N_x1=N_θ, order='leading')
# np.save("./out/egno_leading.npy", leading)
# del leading
# delta = anguelova.calc_delta(args, *extent, N_x0=N_r, N_x1=N_θ)
# np.save("./out/egno_delta.npy", delta)
# del delta
# omega = anguelova.calc_omega(args, *extent, N_x0=N_r, N_x1=N_θ)
# np.save("./out/egno_omega.npy", omega)
# del omega
epsilon = anguelova.calc_epsilon(args, *extent, N_x0=N_r, N_x1=N_θ)
np.save("./out/egno_epsilon.npy", epsilon)
del epsilon

# qdif = anguelova.flag_quantum_dif(args, *extent, accuracy=1e-2)
# np.save("./out/egno_qdif.npy", qdif)
# del qdif

# r_start, r_stop = 0.4, 0.6
# potential = anguelova.calc_V_array(
#   args,
#   [r_start, θ_start],
#   [r_stop, θ_stop],
#   [N_r, N_θ]
# )
# np.save("./out/egno_potential.npy", potential)