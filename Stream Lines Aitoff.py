import numpy as np
import pandas as pd
from tqdm import tqdm
import cmasher as cmr
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
plt.rcParams["text.usetex"] = True

# LOAD ROTATION CURVE
file_path = 'MW Grand Rotation Curve 2021.dat'
df = pd.read_csv(file_path, header = 0, delimiter = r'\s+')
df = df.rename(columns = {'R(kpc),':'R', 'dR(St.dev),':'dR', 'V(km/s),':'V', 'dV(St.dev)':'dV'})
df['R'] = df['R']*1000

# INTERPOLATE
def VelGal(x, R, V):
    tck = interpolate.splrep(R, V)
    return interpolate.splev(x, tck)

def expected_proper_motion(l, b, r, zsun = 20, Usun = 11.1, Vsun = 12.24, Wsun = 7.25, R0 = 8178, Vc = 240):
    yr2sec = 3600*24*365
    pc2km = 3.26156*299792.458*yr2sec
    deg2rad = np.pi/180
    radsec2masyr = (3600*1000/deg2rad)*yr2sec

    l = l*deg2rad # rad
    b = b*deg2rad # rad

    Rc = np.sqrt((R0**2)-(zsun**2))
    s = zsun/R0
    ss = np.sqrt(1-(s**2))
    sl = np.sin(l)
    cl = np.cos(l)
    sb = np.sin(b)
    cb = np.cos(b)

    R = np.sqrt((Rc**2)+((r*cb)**2)-2*ss*Rc*r*cb*cl+(((s*r)**2)*((sb**2)-((cb*cl)**2)))-2*s*Rc*r*sb+2*s*ss*(r**2)*cb*sb*cl)
    V = VelGal(R, df['R'].values, df['V'].values) # km/s
    Vc = VelGal(Rc, df['R'].values, df['V'].values) # km/s
    omega = V/R

    Vr = -Usun*cb*cl+(omega*Rc-Vsun-Vc)*cb*sl-Wsun*sb # km/s
    pml = (1/r)*(Usun*sl+(omega*Rc-Vsun-Vc)*cl-omega*r*(ss*cb+s*sb*cl))*(radsec2masyr/pc2km) # mas/yr
    pmb = (1/r)*(Usun*sb*cl+(Vsun+Vc-omega*Rc)*sb*sl-Wsun*cb+omega*s*r*sl)*(radsec2masyr/pc2km) # mas/yr
    return Vr, pml, pmb

# Grid of x, y points
nx, ny = 200, 100
lon = np.linspace(0, 360, nx)
lat = np.linspace(-90, 90, ny)

# Vector field
u = np.zeros((ny, nx)).astype(np.float32)
v = np.zeros((ny, nx)).astype(np.float32)
vr = np.zeros((ny, nx)).astype(np.float32)
for i in tqdm(range(len(lat)), leave = False, desc = 'Creating vector field'):
    for j in range(len(lon)):
        Vr, pml, pmb = expected_proper_motion(lon[j], lat[i], 3000)
        u[i][j] = pml
        v[i][j] = pmb
        vr[i][j] = Vr

# Center and longitude-reverse vector field
u = -u[:,::-1]
left_u = u[:,:int(np.round(u.shape[1]/2, 0))]
right_u = u[:,int(np.round(u.shape[1]/2, 0)):]
u = np.concatenate([right_u, left_u], axis = 1)
v = v[:,::-1]
left_v = v[:,:int(np.round(v.shape[1]/2, 0))]
right_v = v[:,int(np.round(v.shape[1]/2, 0)):]
v = np.concatenate([right_v, left_v], axis = 1)
vr = vr[:,::-1]
left_vr = vr[:,:int(np.round(vr.shape[1]/2, 0))]
right_vr = vr[:,int(np.round(vr.shape[1]/2, 0)):]
vr = np.concatenate([right_vr, left_vr], axis = 1)

plt.style.use('dark_background')
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection = 'aitoff')
norm = Normalize(vmin=np.min(vr), vmax=np.max(vr))
ax.streamplot(np.deg2rad(lon)-np.pi, np.deg2rad(lat), u, v, color = vr,
              linewidth = 1.5, cmap = 'cmr.guppy_r', density = 2.5,
              arrowstyle = '->', arrowsize = 1.0, zorder = 0, norm = norm)
cbar = fig.colorbar(ScalarMappable(norm=norm, cmap='cmr.guppy_r'), ax=ax,
                    orientation='horizontal', pad = 0.05, fraction = 0.045, aspect = 25)
cbar.set_label('Radial Velocity (km/s)')
ax.grid(True, color = 'grey', alpha = 0.5)
x_tick_labels = np.array([r'$150^{\circ}$', r'$120^{\circ}$', r'$90^{\circ}$',
                          r'$60^{\circ}$', r'$30^{\circ}$', r'$l = 0^{\circ}$',
                          r'$330^{\circ}$', r'$300^{\circ}$', r'$270^{\circ}$',
                          r'$240^{\circ}$', r'$210^{\circ}$'])
ax.set_xticklabels(x_tick_labels, zorder = 15)
y_tick_labels = np.array([r'$-75^{\circ}$', r'$-60^{\circ}$', r'$-45^{\circ}$',
                          r'$-30^{\circ}$', r'$-15^{\circ}$', r'$b = 0^{\circ}$',
                          r'$15^{\circ}$', r'$30^{\circ}$', r'$45^{\circ}$',
                          r'$60^{\circ}$', r'$75^{\circ}$'])
ax.set_yticklabels(y_tick_labels)
plt.savefig('StreamLines_Aitoff_FarField_04.png', format = 'png', dpi = 400, bbox_inches = 'tight', pad_inches = 0)
