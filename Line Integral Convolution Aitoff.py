import numpy as np
import pandas as pd
from tqdm import tqdm
import cmasher as cmr
from PIL import Image
from scipy import interpolate
import matplotlib.pyplot as plt
from licplot import lic_internal
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
    # V = Vc
    omega = V/R

    Vr = -Usun*cb*cl+(omega*Rc-Vsun-Vc)*cb*sl-Wsun*sb # km/s
    pml = (1/r)*(Usun*sl+(omega*Rc-Vsun-Vc)*cl-omega*r*(ss*cb+s*sb*cl))*(radsec2masyr/pc2km) # mas/yr
    pmb = (1/r)*(Usun*sb*cl+(Vsun+Vc-omega*Rc)*sb*sl-Wsun*cb+omega*s*r*sl)*(radsec2masyr/pc2km) # mas/yr
    return Vr, pml, pmb

# Grid of x, y points
nx, ny = 2000, 1000
lon = np.linspace(0, 360, nx)
lat = np.linspace(-90, 90, ny)

# Vector field
u = np.zeros((ny, nx)).astype(np.float32)
v = np.zeros((ny, nx)).astype(np.float32)
for i in tqdm(range(len(lat)), leave = False, desc = 'Creating vector field'):
    for j in range(len(lon)):
        Vr, pml, pmb = expected_proper_motion(lon[j], lat[i], 3000)
        u[i][j] = pml
        v[i][j] = pmb

# Center and longitude-reverse vector field
u = -u[:,::-1]
left_u = u[:,:int(np.round(u.shape[1]/2, 0))]
right_u = u[:,int(np.round(u.shape[1]/2, 0)):]
u = np.concatenate([right_u, left_u], axis = 1)
v = v[:,::-1]
left_v = v[:,:int(np.round(v.shape[1]/2, 0))]
right_v = v[:,int(np.round(v.shape[1]/2, 0)):]
v = np.concatenate([right_v, left_v], axis = 1)

# Create texture
np.random.seed(15)
texture = np.random.choice([0, 1], size = (100, 200)).astype(np.float32)

# Upscale texture
upscaled_texture = Image.fromarray(texture).resize([nx, ny], resample = Image.NEAREST)
upscaled_texture = np.asarray(upscaled_texture)

# Line Integral Convolution
kernel_length = 500
kernel = np.sin((np.arange(kernel_length) * np.pi / kernel_length)).astype(np.float32)
image = lic_internal.line_integral_convolution(u, v, upscaled_texture, kernel)
plt.clf()
plt.axis("off")
plt.imshow(image[:,::-1], cmap = 'cmr.wildfire', interpolation = 'gaussian')
plt.savefig('LIC_FarField_04.png', format = 'png', dpi = 400, bbox_inches = 'tight', pad_inches = 0)
plt.close('all')

# Aitoff projection
x = np.linspace(-np.pi, np.pi, image.shape[1])
y = np.linspace(-np.pi/2, np.pi/2, image.shape[0])
X,Y = np.meshgrid(x,y)
plt.style.use('dark_background')
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot(111, projection = 'aitoff')
plt.pcolormesh(X, Y, image, cmap = 'cmr.wildfire')
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
fig.savefig('LIC_FarField_Aitoff_04.png', format = 'png', dpi = 400, bbox_inches = 'tight', pad_inches = 0)
