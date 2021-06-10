import numpy as np
import glob
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from numba import jit
import numba
from random import randrange
import random
import pandas as pd
from matplotlib.cm import ScalarMappable

L = 10
Lx = 3 * L
Ly = 1 * L

nr_p = 150

# the diffusion and interaction range settings -----

sigma = 0.0005			# controls the diffusional noise
delta = 0.1				# max distance per step - delta has to be smaller than dmax / 2 !

# the interaction potential parameters -----

u0 = -1.0				# interaction pre-factor
T = 1.0					# dimensionless temperature
gamma = 1.0 			# particle diameter
dmax = 10.0 * gamma 	# max. interaction range
intexp = 200			# intraction exponent

for i in range(1,100):
	print (round(0.05*i,5), round(u0 * (gamma/(0.05*i))**intexp,5))

mcmax = nr_p * 10000

lat = np.asarray(np.zeros([nr_p,2]), dtype='f8')
vlat = np.asarray(np.zeros([nr_p,2]), dtype='f8')

particlesonlattice = 1

lat[0,0] = round(Lx * np.random.rand(),5)
lat[0,1] = round(Ly * np.random.rand(),5)
vlat[0,0] = round(delta * np.random.rand() * random.choice((-1, 1)),5)
vlat[0,1] = round(delta * np.random.rand() * random.choice((-1, 1)),5)

while particlesonlattice < nr_p:
	x_new = round(Lx * np.random.rand(),5)
	y_new = round(Ly * np.random.rand(),5)
	overlap = 0
	for i in range(nr_p):
		if lat[i,0] != 0:
			dist = np.sqrt((x_new - lat[i,0])**2 + (y_new - lat[i,1])**2)
			if dist < gamma:
				overlap = 1
	if overlap == 0:
		lat[particlesonlattice,0] = x_new
		lat[particlesonlattice,1] = y_new
		vlat[particlesonlattice,0] = round(delta * np.random.rand() * random.choice((-1, 1)),5)
		vlat[particlesonlattice,1] = round(delta * np.random.rand() * random.choice((-1, 1)),5)
		particlesonlattice += 1

@numba.njit
def lattice_mips(mcmax,sigma,nr_p,delta,Lx,Ly,lat,vlat):
	MCS = 0
	for i in range(mcmax):
		MCS += 1
		rndidx = np.random.randint(len(lat))
		x, y = lat[rndidx,0], lat[rndidx,1]
		vx, vy = vlat[rndidx,0], vlat[rndidx,1]
		x_new, y_new = x + vx, y + vy
		if x_new >= Lx:
			x_new -= Lx
		if x_new < 0:
			x_new += Lx
		if y_new >= Ly:
			y_new -= Ly
		if y_new < 0:
			y_new += Ly
		vx_new = np.random.normal(vx, sigma, 1)[0]
		vy_new = np.random.normal(vy, sigma, 1)[0]
		if vx_new >= delta:
			vx_new = delta - (vx_new - delta)
		if vx_new <= -delta:
			vx_new = -delta - (vx_new + delta)
		if vy_new >= delta:
			vy_new = delta - (vy_new - delta)
		if vy_new <= -delta:
			vy_new = -delta - (vy_new + delta)
		U_t, U_t1 = 0, 0
		overlap = 0
		for j in range(nr_p):
			dx = round(abs(x-lat[j,0]),5)
			dy = round(abs(y-lat[j,1]),5)
			dx_new = round(abs(x_new-lat[j,0]),5)
			dy_new = round(abs(y_new-lat[j,1]),5)
			if dx >= Lx/2:
				dx -= Lx
			if dx_new >= Lx/2:
				dx_new -= Lx
			if dy >= Ly/2:
				dy -= Ly
			if dy_new >= Ly/2:
				dy_new -= Ly
			dr = round(np.sqrt(dx**2 + dy**2),5)
			dr_new = round(np.sqrt(dx_new**2 + dy_new**2),5)
			if dr_new < gamma and lat[j,0] != x:
				overlap = 1
			if dr != 0:
				e_x, e_y = (lat[j,0] - x), (lat[j,1] - y)
				e_x_new, e_y_new = (lat[j,0] - x_new), (lat[j,1] - y_new)
				if e_x >= Lx/2:
					e_x -= Lx
				if e_x < 0:
					e_x += Lx
				if e_y >= Ly/2:
					e_y -= Ly
				if e_y < 0:
					e_y += Ly
				if e_x_new >= Lx/2:
					e_x_new -= Lx
				if e_x_new < 0:
					e_x_new += Lx
				if e_y_new >= Ly/2:
					e_y_new -= Ly
				if e_y_new < 0:
					e_y_new += Ly
				if dr <= dmax:
					U_t += u0 * (gamma/dr)**intexp
				if dr_new <= dmax:
					U_t1 += u0 * (gamma/dr_new)**intexp
		move = 0
		if U_t1 <= U_t and overlap == 0:
			move = 1
			lat[rndidx,0] = x_new
			lat[rndidx,1] = y_new
			vlat[rndidx,0] = vx_new
			vlat[rndidx,1] = vy_new
		else:
			if np.random.rand() <= np.exp(-(U_t1-U_t)/T) and overlap == 0:
				move = 1
				lat[rndidx,0] = x_new
				lat[rndidx,1] = y_new
				vlat[rndidx,0] = vx_new
				vlat[rndidx,1] = vy_new
		if move == 0:
			vlat[rndidx,0] = vx_new
			vlat[rndidx,1] = vy_new
		if int(MCS/(10*nr_p)) == MCS/(10*nr_p):
			print (MCS/nr_p)

print ('# persistence time:', 8/np.pi**2 * delta**2/sigma**3)
print ('# persistence length:', 0.62 * delta**3 / sigma**2)
print ('# particle density:', gamma**2 * nr_p / (Lx*Ly))

lattice_mips(mcmax,sigma,nr_p,delta,Lx,Ly,lat,vlat)

lat_x = []
lat_y = []
lat_c = []

for i in range(nr_p):
	lat_x.append(lat[i,0])
	lat_y.append(lat[i,1])
	o_x, o_y = vlat[i,0], vlat[i,1]
	r_x, r_y = 1, 0
	if o_y >= 0:
		#print ('POSITIVE', o_x, o_y)
		#print (np.arccos((o_x * r_x + o_y * r_y) / (np.sqrt(o_x**2 + o_y**2) * np.sqrt(r_x**2 + r_y**2))) * (180/np.pi))
		lat_c.append(np.arccos((o_x * r_x + o_y * r_y) / (np.sqrt(o_x**2 + o_y**2) * np.sqrt(r_x**2 + r_y**2))) * (180/np.pi))
	if o_y < 0:
		#print (360 - np.arccos((o_x * r_x + o_y * r_y) / (np.sqrt(o_x**2 + o_y**2) * np.sqrt(r_x**2 + r_y**2))) * (180/np.pi))
		#print (vlat[i,0], o_x, vlat[i,1], o_y)
		lat_c.append(360 - np.arccos((o_x * r_x + o_y * r_y) / (np.sqrt(o_x**2 + o_y**2) * np.sqrt(r_x**2 + r_y**2))) * (180/np.pi))

fig, axes = plt.subplots(1,1, constrained_layout=False, sharey=True)
axes.set_aspect(1)
fig.canvas.draw()
s = 60# ((axes.get_window_extent().width  / (L-0+1.) * 72./fig.dpi) ** 2)
axes.scatter(lat_x, lat_y, s = s, c=lat_c, cmap='twilight', linewidth=0)

cmap = plt.get_cmap("twilight")
scales = np.linspace(0, 360, 10)
norm = plt.Normalize(scales.min(), scales.max())
sm =  ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes)
cbar.ax.set_title("scale")

plt.show()
