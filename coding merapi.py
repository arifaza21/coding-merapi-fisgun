import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Merapi.txt')
data

x = data[:, 0]; y = data[:, 1]; z = data[:, 2]

xm = x.reshape(25, 19); ym = y.reshape(25, 19)
zm = z.reshape(25, 19)

plt.pcolormesh(xm, ym, zm, cmap = 'jet')
plt.colorbar()

xm = x.reshape(25, 19); ym = y.reshape(25, 19)
zm = z.reshape(25, 19)

plt.contourf(xm, ym, zm, cmap = 'jet')
plt.colorbar()

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(xm, ym, zm, cmap = 'jet')

dx = np.random.rand(25, 19)
dy = np.random.rand(25, 19)
a = 2.5; f = 2; p = 0.5; miu = 0.25

d = np.zeros ((25, 19), float)
delta_d = np.zeros((25, 19), float)
delta_h = np.zeros((25, 19), float)
for i in range(0, 25):
    for j in range(0, 19):
        d[i,j] = np.sqrt(dx[i,j] ** 2 + dy[i,j] ** 2)
        delta_d[i,j] = (3 * a ** 3 * p * d[i,j]) / (4 * miu * (f ** 2 + d[i,j] ** 2) ** 1.5)
        delta_h[i,j] = (3 * a ** 3 * p * f) / (4 * miu * (f ** 2 + d[i,j] ** 2) ** 1.5)

theta = np.zeros((25,19), float)
Dx = np.zeros((25,19), float)
Dy = np.zeros((25,19), float)
for i in range (0,25):
  for j in range (0,19):
    theta[i,j] = np.arctan2(dy[i,j],dx[i,j])
    Dx[i,j] = delta_d[i,j]*np.cos(theta[i,j])
    Dy[i,j] = delta_d[i,j]*np.sin(theta[i,j])

xb = xm + Dx
yb = ym + Dy
zb = zm + delta_h

ax.plot_surface(xb, yb, zb)

