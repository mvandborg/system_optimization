
# %% imports

from sympy import *
init_printing(use_unicode=False,wrap_line=False)

# %% Integration of lorentzian

G = Symbol('\Gamma',real=True,positive=True)
w = Symbol('\omega',real=True,positive=True)
lor = (G/2)**2/(w**2+(G/2)**2)

I = integrate(lor,(w,-oo,oo))

# %% Fourier transform of exponentiao
t = Symbol('t',real=True)

omega = Symbol('\omega',real=True)

FT = integrate(exp(G*abs(t))*exp(1j*omega*t),(t,-oo,oo))

# %%
