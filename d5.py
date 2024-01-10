#Script to quickly generate data for d5-brane inflation model using inflatox
import sympy
from sympy.simplify.radsimp import collect_sqrt
import numpy as np
import inflatox

print("[D5-brane string theory inflation script]")
model = "d5"

################################################################################
#                                 model set-up                                 #
################################################################################

r, θ = sympy.symbols('r θ2')
fields = [r, θ]

gs, ls, N = sympy.symbols('g_s l_s N')
mu5, T5, Lt = sympy.symbols('mu5 T5 L_T')

mu5 = 1/((2 * sympy.pi)**5 * ls**6)
T5 = mu5 / gs

rho, u = sympy.symbols('rho u')
rho = r / (3*u)

H = (
  (sympy.pi*N*gs*ls**4)/(12*u**4) * (2/rho**2 - 2*sympy.ln(1/rho**2 +1))
).nsimplify().collect([u, r]).expand().powsimp(force=True)

p, q = sympy.symbols('p q')

F = (
  H / 9 * (r**2 + 3*u**2)**2 + (sympy.pi*q*ls**2)**2
).nsimplify().collect([r, u]).expand().powsimp()

gamma = 4*sympy.pi**2*ls**2*p*q*T5*gs

sqrtF = sympy.sqrt(F)
g00 = collect_sqrt(
  4*sympy.pi*p*T5 * sqrtF * ((r**2+6*u**2)/(r**2+p*u**2)),
evaluate=True).expand().powsimp()
g11 = collect_sqrt(
  (4/6) * sympy.pi*p*T5 * sqrtF * (r**2+6*u**2),
evaluate=True).nsimplify().collect([r, u]).expand().powsimp()

metric = [
  [g00, 0],
  [0, g11]
]

Phi_min = ( (5/72) * (
  81*(9*rho**2 - 2)*rho**2 +
  162*sympy.ln(9*(rho**2 + 1)) +
  -9 +
  -160*sympy.ln(10)
)).nsimplify().collect([u]).expand().powsimp()

a0, a1, b1 = sympy.symbols('a0 a1 b1')
Phi_h = (a0 * (2/rho**2 -2*sympy.ln(1/rho**2 + 1))
  + 2*a1*(
    6 + 1/rho**2 -2*(2+3*rho**2)*sympy.ln(1 + 1/rho**2)
  )*sympy.cos(θ)
  + (b1/2) * (2+3*rho**2)*sympy.cos(θ)
).nsimplify().collect([u, r]).expand().powsimp()

V0 = sympy.symbols('V0')
potential = V0 + (4*sympy.pi*p*T5/H) * (sympy.sqrt(F)-(ls**2)*sympy.pi*q*gs) + gamma*(Phi_min + Phi_h)
potential = potential.nsimplify().collect([ls, gs]).expand().powsimp()

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
  silent=True
).execute([[0,1]])

out = inflatox.Compiler(hesse).compile()
out.print_sym_lookup_table()

################################################################################
#                                   paramters                                  #
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

#run Anguelova's original condition
consistency_old = anguelova.consistency_only_old(
  parameters,
  *extent,
  *[N, N]
)
np.save(f"./out/{model}_old.npy", consistency_old)