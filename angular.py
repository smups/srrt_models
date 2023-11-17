#Script to quickly generate data for angular inflation model using inflatox
import sympy
import numpy as np
import inflatox

print("[Angular inflation script]")

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
potential = anguelova.calc_V_array(
  args,
  [phi_start, chi_start],
  [phi_stop, chi_stop],
  [N, N]
)
np.save("./out/angular_potential.npy", potential)
del potential

exact = anguelova.evaluate(args, *extent, order='exact')
np.save("./out/angular_exact.npy", exact)
del exact
leading = anguelova.evaluate(args, *extent, order='leading')
np.save("./out/angular_leading.npy", leading)
del leading
delta = anguelova.calc_delta(args, *extent)
np.save("./out/angular_delta.npy", delta)
del delta
omega = anguelova.calc_omega(args, *extent)
np.save("./out/angular_omega.npy", omega)
del omega

qdif = anguelova.flag_quantum_dif(args, *extent, accuracy=1e-2)
np.save("./out/angular_qdif.npy", qdif)