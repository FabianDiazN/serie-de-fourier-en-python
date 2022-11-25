import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import scipy as sp
import sympy as sym

n = sym.Symbol('n')
t = sym.Symbol('t')

# funcion periodica
Tmin = 0
Tmax = 2*np.pi

T = Tmax-Tmin
w = 2*np.pi/T

# ft es una función simbólica
ft = t

f_integral = ft
a0 = (2/T)*sym.integrate(f_integral, (t, Tmin, Tmax))
print("a0 = ")
sym.pprint(a0)

# Calculamos los coeficientes de Fourier

# Calculamos la integral para an
f_integral = ft*sym.cos(n*w*t)
an = (2/T)*sym.integrate(f_integral, (t, Tmin, Tmax))
an = sym.simplify(an)
print("an = ")
sym.pprint(an)

# Calculamos la integral para bn
f_integral = ft*sym.sin(n*w*t)
bn = (2/T)*sym.integrate(f_integral, (t, Tmin, Tmax))
print("bn = ")
bn = sym.simplify(bn)
sym.pprint(bn)

# Definimos el número de armónicos para la expansión
serie = 0
Armonicos = 30

for i in range(1, Armonicos+1):

    # Evaluamos los coeficientes para cada armónico
    an_c = an.subs(n, i)
    bn_c = bn.subs(n, i)

    if abs(an_c) < 0.0001:
        an_c = 0
    if abs(bn_c) < 0.0001:
        bn_c = 0

    serie = serie + an_c*sym.cos(i*w*t)  # Términos coseno de la serie
    serie = serie + bn_c*sym.sin(i*w*t)  # Términos seno de la serie

serie = a0/2+serie  # Expansión final de la serie

print('f(t)= ')
sym.pprint(serie)
# Convertimos la expresión Sympy a una función evaluable
fserie = sym.lambdify(t, serie)
f = sym.lambdify(t, ft)

# Creamos un vector de tiempo para la gráfica
v_tiempo = np.linspace(Tmin, Tmax, 200)

# Evaluamos las funciones
fserieG = fserie(v_tiempo)
fG = f(v_tiempo)

plt.plot(v_tiempo, fG, label='f(t)')
plt.plot(v_tiempo, fserieG, label='Expansión')

plt.xlabel('tiempo')
plt.legend()
plt.title('Expansión en Series de Fourier')
plt.show()

# Función por tramos

n = sym.Symbol('n')
t = sym.Symbol('t')

# Definimos la función peridódica por tramos
Tmin = -3
Tmax = 3

T = Tmax-Tmin
w = 2*np.pi/T

f1 = (2/3)*t+2
f2 = -(2/3)*t+2

# ft es una función simbólica por tramos
ft = sym.Piecewise((f1, ((t <= 0) & (t >= -3))), (f2, ((t > 0) & (t <= 3))))
ft

# Calculamos la integral para a0
f_integral = ft
a0 = (2/T)*sym.integrate(f_integral, (t, Tmin, Tmax))
print("a0 = ")
sym.pprint(a0)

# Calculamos la integral para an
f_integral = ft*sym.cos(n*w*t)
an = (2/T)*sym.integrate(f_integral, (t, Tmin, Tmax))
an = sym.simplify(an)
print("an = ")
sym.pprint(an)

# Calculamos la integral para bn
f_integral = ft*sym.sin(n*w*t)
bn = (2/T)*sym.integrate(f_integral, (t, Tmin, Tmax))
print("bn = ")
bn = sym.simplify(bn)
sym.pprint(bn)
# Usando los coeficientes representamos la expansión en SF

# Definimos el número de armónicos para la expansión
serie = 0
Armonicos = 30

for i in range(1, Armonicos+1):

    # Evaluamos los coeficientes para cada armónico
    an_c = an.subs(n, i)
    bn_c = bn.subs(n, i)

    if abs(an_c) < 0.0001:
        an_c = 0
    if abs(bn_c) < 0.0001:
        bn_c = 0

    serie = serie + an_c*sym.cos(i*w*t)  # Términos coseno de la serie
    serie = serie + bn_c*sym.sin(i*w*t)  # Términos seno de la serie

serie = a0/2+serie  # Expansión final de la serie

print('f(t)= ')
sym.pprint(serie)

# Graficamos la función periódica original y su expansión en Series de Fourier

# Convertimos la expresión Sympy a una función evaluable
fserie = sym.lambdify(t, serie)
f = sym.lambdify(t, ft)

# Creamos un vector de tiempo para la gráfica
v_tiempo = np.linspace(Tmin, Tmax, 200)

# Evaluamos las funciones
fserieG = fserie(v_tiempo)
fG = f(v_tiempo)

plt.plot(v_tiempo, fG, label='f(t)')
plt.plot(v_tiempo, fserieG, label='Expansión')

plt.xlabel('tiempo')
plt.legend()
plt.title('Expansión en Series de Fourier')
plt.show()
