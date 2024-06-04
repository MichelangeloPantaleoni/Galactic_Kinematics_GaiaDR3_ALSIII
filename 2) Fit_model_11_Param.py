import numpy as np
import pandas as pd
from tqdm import tqdm
import cmasher as cmr
from scipy import interpolate
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
plt.rcParams["text.usetex"] = True

def round_to_significant_digits(value, uncertainty, sig_digits = 2):
    order_of_magnitude = np.floor(np.log10(uncertainty))
    factor = 10**(sig_digits - 1 - order_of_magnitude)
    rounded_value = np.round(value * factor) / factor
    rounded_uncertainty = np.round(uncertainty * factor) / factor
    return rounded_value, rounded_uncertainty

def patch_array(x, y, x_patch, y_patch, start_val, end_val):
    x_start_idx = np.searchsorted(x, start_val)
    x_end_idx = np.searchsorted(x, end_val)
    x_patch_start_idx = np.searchsorted(x_patch, start_val)
    x_patch_end_idx = np.searchsorted(x_patch, end_val)
    x_piece1 = x[:x_start_idx]
    x_piece2 = x_patch[x_patch_start_idx:x_patch_end_idx+1]
    x_piece3 = x[x_end_idx:]
    y_piece1 = y[:x_start_idx]
    y_piece2 = y_patch[x_patch_start_idx:x_patch_end_idx+1]
    y_piece3 = y[x_end_idx:]
    x_patched = np.concatenate([x_piece1, x_piece2, x_piece3])
    y_patched = np.concatenate([y_piece1, y_piece2, y_piece3])
    return x_patched, y_patched

def VelGal(x, R, V, start_patch = None, end_patch = None):
    if not (isinstance(x, np.ndarray) or isinstance(x, list)):
        x = np.array([x])
        single_val = True
    else:
        single_val = False
    V_interp = []
    for i in range(len(x)):
        if (start_patch is not None) and (x[i] >= start_patch) and (x[i] <= end_patch):
            # Linear interpolation
            V_linear = np.interp(x[i], R, V)
            V_interp.append(V_linear)
        else:
            # Spline interpolation
            tck = interpolate.splrep(R, V)
            V_spline = interpolate.splev(x[i], tck)
            V_interp.append(V_spline)
    if single_val:
        return V_interp[0]
    else:
        return np.array(V_interp)

def model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, show_more = False):
    global R_rot_original, V_rot_original, R_patch, R_range_min, R_range_max
    V_patch = np.array([V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11])
    R_rot, V_rot = patch_array(R_rot_original, V_rot_original, R_patch, V_patch, R_range_min, R_range_max)

    yr2sec = 3600*24*365
    pc2km = 3.26156*299792.458*yr2sec
    deg2rad = np.pi/180
    radsec2masyr = (3600*1000/deg2rad)*yr2sec

    l, b, r = X
    l = l*deg2rad # rad
    b = b*deg2rad # rad
    R0 = 8178 # pc
    Rc = np.sqrt((R0**2)-(zsun**2)) # pc
    s = zsun/R0
    ss = np.sqrt(1-(s**2))
    sl = np.sin(l)
    cl = np.cos(l)
    sb = np.sin(b)
    cb = np.cos(b)

    R = np.sqrt((Rc**2)+((r*cb)**2)-2*ss*Rc*r*cb*cl+(((s*r)**2)*((sb**2)-((cb*cl)**2)))-2*s*Rc*r*sb+2*s*ss*(r**2)*cb*sb*cl)
    V = VelGal(R, R_rot, V_rot, R_range_min, R_range_max) # km/s
    Vc = VelGal(Rc, R_rot, V_rot, R_range_min, R_range_max)
    omega = V/R

    pml = (1/r)*(Usun*sl+(omega*Rc-Vsun-Vc)*cl-omega*r*(ss*cb+s*sb*cl))*(radsec2masyr/pc2km) # mas/yr
    pmb = (1/r)*(Usun*sb*cl+(Vsun+Vc-omega*Rc)*sb*sl-Wsun*cb+omega*s*r*sl)*(radsec2masyr/pc2km) # mas/yr

    if show_more:
        Vr = -Usun*cb*cl+(omega*Rc-Vsun-Vc)*cb*sl-Wsun*sb
        return R, Vr, pml, pmb
    else:
        return pml, pmb

def sigma_model(l, b, r, zsun, Usun, Vsun, Wsun, sigma_zsun, sigma_Usun, sigma_Vsun, sigma_Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, sigma_V1, sigma_V2, sigma_V3, sigma_V4, sigma_V5, sigma_V6, sigma_V7, sigma_V8, sigma_V9, sigma_V10, sigma_V11):
    yr2sec = 3600*24*365
    pc2km = 3.26156*299792.458*yr2sec
    deg2rad = np.pi/180
    radsec2masyr = (3600*1000/deg2rad)*yr2sec

    X = (l, b, r)
    eps_zsun = 0.001 # pc
    eps_Usun, eps_Vsun, eps_Wsun = 0.0001, 0.0001, 0.0001 # km/s
    eps_V = 0.001 # km/s

    pml, pmb = model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)
    dpml_dzsun = (model(X, zsun+eps_zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)[0]-pml)/eps_zsun
    dpml_dUsun = (model(X, zsun, Usun+eps_Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)[0]-pml)/eps_Usun
    dpml_dVsun = (model(X, zsun, Usun, Vsun+eps_Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)[0]-pml)/eps_Vsun
    dpml_dWsun = (model(X, zsun, Usun, Vsun, Wsun+eps_Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)[0]-pml)/eps_Wsun
    dpml_dV1 = (model(X, zsun, Usun, Vsun, Wsun, V1+eps_V, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)[0]-pml)/eps_V
    dpml_dV2 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2+eps_V, V3, V4, V5, V6, V7, V8, V9, V10, V11)[0]-pml)/eps_V
    dpml_dV3 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3+eps_V, V4, V5, V6, V7, V8, V9, V10, V11)[0]-pml)/eps_V
    dpml_dV4 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4+eps_V, V5, V6, V7, V8, V9, V10, V11)[0]-pml)/eps_V
    dpml_dV5 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5+eps_V, V6, V7, V8, V9, V10, V11)[0]-pml)/eps_V
    dpml_dV6 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6+eps_V, V7, V8, V9, V10, V11)[0]-pml)/eps_V
    dpml_dV7 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7+eps_V, V8, V9, V10, V11)[0]-pml)/eps_V
    dpml_dV8 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8+eps_V, V9, V10, V11)[0]-pml)/eps_V
    dpml_dV9 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9+eps_V, V10, V11)[0]-pml)/eps_V
    dpml_dV10 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10+eps_V, V11)[0]-pml)/eps_V
    dpml_dV11 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11+eps_V)[0]-pml)/eps_V

    dpmb_dzsun = (model(X, zsun+eps_zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)[1]-pmb)/eps_zsun
    dpmb_dUsun = (model(X, zsun, Usun+eps_Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)[1]-pmb)/eps_Usun
    dpmb_dVsun = (model(X, zsun, Usun, Vsun+eps_Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)[1]-pmb)/eps_Vsun
    dpmb_dWsun = (model(X, zsun, Usun, Vsun, Wsun+eps_Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)[1]-pmb)/eps_Wsun
    dpmb_dV1 = (model(X, zsun, Usun, Vsun, Wsun, V1+eps_V, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11)[1]-pmb)/eps_V
    dpmb_dV2 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2+eps_V, V3, V4, V5, V6, V7, V8, V9, V10, V11)[1]-pmb)/eps_V
    dpmb_dV3 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3+eps_V, V4, V5, V6, V7, V8, V9, V10, V11)[1]-pmb)/eps_V
    dpmb_dV4 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4+eps_V, V5, V6, V7, V8, V9, V10, V11)[1]-pmb)/eps_V
    dpmb_dV5 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5+eps_V, V6, V7, V8, V9, V10, V11)[0]-pmb)/eps_V
    dpmb_dV6 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6+eps_V, V7, V8, V9, V10, V11)[0]-pmb)/eps_V
    dpmb_dV7 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7+eps_V, V8, V9, V10, V11)[0]-pmb)/eps_V
    dpmb_dV8 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8+eps_V, V9, V10, V11)[0]-pmb)/eps_V
    dpmb_dV9 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9+eps_V, V10, V11)[0]-pmb)/eps_V
    dpmb_dV10 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10+eps_V, V11)[0]-pmb)/eps_V
    dpmb_dV11 = (model(X, zsun, Usun, Vsun, Wsun, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11+eps_V)[0]-pmb)/eps_V

    fl = np.array([dpml_dzsun*sigma_zsun, dpml_dUsun*sigma_Usun, dpml_dVsun*sigma_Vsun, dpml_dWsun*sigma_Wsun,
                   dpml_dV1*sigma_V1, dpml_dV2*sigma_V2, dpml_dV3*sigma_V3, dpml_dV4*sigma_V4,
                   dpml_dV5*sigma_V5, dpml_dV6*sigma_V6, dpml_dV7*sigma_V7, dpml_dV8*sigma_V8,
                   dpml_dV9*sigma_V9, dpml_dV10*sigma_V10, dpml_dV11*sigma_V11])
    fb = np.array([dpmb_dzsun*sigma_zsun, dpmb_dUsun*sigma_Usun, dpmb_dVsun*sigma_Vsun, dpmb_dWsun*sigma_Wsun,
                   dpmb_dV1*sigma_V1, dpmb_dV2*sigma_V2, dpmb_dV3*sigma_V3, dpmb_dV4*sigma_V4,
                   dpmb_dV5*sigma_V5, dpmb_dV6*sigma_V6, dpmb_dV7*sigma_V7, dpmb_dV8*sigma_V8,
                   dpmb_dV9*sigma_V9, dpmb_dV10*sigma_V10, dpmb_dV11*sigma_V11])
    sig_pml_th = np.sqrt(np.sum(fl**2))
    sig_pmb_th = np.sqrt(np.sum(fb**2))
    return sig_pml_th, sig_pmb_th

# LOAD ROTATION CURVE
file_path = 'MW Grand Rotation Curve 2021.dat'
df_rot = pd.read_csv(file_path, header = 0, delimiter = r'\s+')
df_rot = df_rot.rename(columns = {'R(kpc),':'R', 'dR(St.dev),':'dR', 'V(km/s),':'V', 'dV(St.dev)':'dV'})
df_rot['R'] = df_rot['R']*1000
df_rot['dR'] = df_rot['dR']*1000
R_rot_original, V_rot_original = df_rot['R'].values, df_rot['V'].values

# INSERT ROTATION CURVE PATCH
R_range_min = 5400 # pc
R_range_max = 10800 # pc
num_rot_curve_params = 11
R_patch = np.linspace(R_range_min, R_range_max, num_rot_curve_params)
V_patch = VelGal(R_patch, R_rot_original, V_rot_original)
num_rot_params = len(R_patch)
R_rot, V_rot = patch_array(R_rot_original, V_rot_original, R_patch, V_patch, R_range_min, R_range_max)

# LOAD SAMPLE DATA
df = pd.read_pickle('ALS_Sample.pkl')
# df = df[(df['parallax_error']/df['parallax'] < 0.2) & (df['Dist_P50'] > 1000)]
# df = df.reset_index(drop = True)
l, b, r = df['l'].values, df['b'].values, df['Dist_P50'].values
pml, pmb = df['pml'].values, df['pmb'].values
pml_error, pmb_error = df['pml_error'].values, df['pmb_error'].values

# SIGMA CLIPPING PARAMETERS
num_its = 15
delta_clip = 40

# INITIALIZE PARAMETER ARRAYS
Zsun_arr, Usun_arr, Vsun_arr, Wsun_arr = np.zeros((4, num_its+1))
Zsun_arr[0] = 20
Usun_arr[0] = 11.1
Vsun_arr[0] = 12.24
Wsun_arr[0] = 7.25

Zsun_err_arr, Usun_err_arr, Vsun_err_arr, Wsun_err_arr = np.zeros((4, num_its+1))
Zsun_err_arr[0] = np.nan
Usun_err_arr[0] = np.nan
Vsun_err_arr[0] = np.nan
Wsun_err_arr[0] = np.nan

V1_arr, V2_arr, V3_arr, V4_arr, V5_arr, V6_arr, V7_arr, V8_arr, V9_arr, V10_arr, V11_arr = np.zeros((num_rot_curve_params, num_its+1))
V1_arr[0] = V_patch[0]
V2_arr[0] = V_patch[1]
V3_arr[0] = V_patch[2]
V4_arr[0] = V_patch[3]
V5_arr[0] = V_patch[4]
V6_arr[0] = V_patch[5]
V7_arr[0] = V_patch[6]
V8_arr[0] = V_patch[7]
V9_arr[0] = V_patch[8]
V10_arr[0] = V_patch[9]
V11_arr[0] = V_patch[10]

V1_err_arr, V2_err_arr, V3_err_arr, V4_err_arr, V5_err_arr, V6_err_arr, V7_err_arr, V8_err_arr, V9_err_arr, V10_err_arr, V11_err_arr = np.zeros((num_rot_curve_params, num_its+1))
V1_err_arr[0] = np.nan
V2_err_arr[0] = np.nan
V3_err_arr[0] = np.nan
V4_err_arr[0] = np.nan
V5_err_arr[0] = np.nan
V6_err_arr[0] = np.nan
V7_err_arr[0] = np.nan
V8_err_arr[0] = np.nan
V9_err_arr[0] = np.nan
V10_err_arr[0] = np.nan
V11_err_arr[0] = np.nan

num_stars_used = []
indx = np.arange(len(df))
for it in tqdm(range(num_its), leave = False, desc = 'Fitting model iteratively'):
    print(f'\n  Using {len(indx)} stars ({100*len(indx)/len(df):.1f} % of the sample)\n')

    l_it, b_it, r_it = l[indx], b[indx], r[indx]
    pml_it, pmb_it = pml[indx], pmb[indx]
    pml_error_it, pmb_error_it = pml_error[indx], pmb_error[indx]

    # FIT MODEL TO DATA
    # Model
    lm_model = Model(model)

    # Parameters (initial values)
    params = Parameters()
    params.add('zsun', value = Zsun_arr[it], vary = False)
    params.add('Usun', value = Usun_arr[it])
    params.add('Vsun', value = Vsun_arr[it])
    params.add('Wsun', value = Wsun_arr[it])
    params.add('V1', value = V1_arr[it])
    params.add('V2', value = V2_arr[it])
    params.add('V3', value = V3_arr[it])
    params.add('V4', value = V4_arr[it])
    params.add('V5', value = V5_arr[it])
    params.add('V6', value = V6_arr[it])
    params.add('V7', value = V7_arr[it])
    params.add('V8', value = V8_arr[it])
    params.add('V9', value = V9_arr[it])
    params.add('V10', value = V10_arr[it])
    params.add('V11', value = V11_arr[it])

    # Fit data
    result = lm_model.fit(data = (pml_it, pmb_it),
                          X = (l_it, b_it, r_it),
                          params = params,
                          weights = (1/pml_error_it, 1/pmb_error_it))

    # Fitted parameters
    print(result.fit_report())
    zsun_best = result.params['zsun'].value
    Usun_best = result.params['Usun'].value
    Vsun_best = result.params['Vsun'].value
    Wsun_best = result.params['Wsun'].value
    V1_best = result.params['V1'].value
    V2_best = result.params['V2'].value
    V3_best = result.params['V3'].value
    V4_best = result.params['V4'].value
    V5_best = result.params['V5'].value
    V6_best = result.params['V6'].value
    V7_best = result.params['V7'].value
    V8_best = result.params['V8'].value
    V9_best = result.params['V9'].value
    V10_best = result.params['V10'].value
    V11_best = result.params['V11'].value

    zsun_best_err = result.params['zsun'].stderr
    Usun_best_err = result.params['Usun'].stderr
    Vsun_best_err = result.params['Vsun'].stderr
    Wsun_best_err = result.params['Wsun'].stderr
    V1_best_err = result.params['V1'].stderr
    V2_best_err = result.params['V2'].stderr
    V3_best_err = result.params['V3'].stderr
    V4_best_err = result.params['V4'].stderr
    V5_best_err = result.params['V5'].stderr
    V6_best_err = result.params['V6'].stderr
    V7_best_err = result.params['V7'].stderr
    V8_best_err = result.params['V8'].stderr
    V9_best_err = result.params['V9'].stderr
    V10_best_err = result.params['V10'].stderr
    V11_best_err = result.params['V11'].stderr
    params_cov_matrix = result.covar

    # DEVIATIONS FROM THE MODEL
    Sig_pml_th, Sig_pmb_th, Delta_pml, Delta_pmb, Delta = [], [], [], [], []
    pml_th_arr, pmb_th_arr = [], []
    for i in range(len(df)):
        pml_th, pmb_th = model((l[i], b[i], r[i]), zsun_best, Usun_best, Vsun_best, Wsun_best, V1_best, V2_best, V3_best, V4_best, V5_best, V6_best, V7_best, V8_best, V9_best, V10_best, V11_best)
        pml_th_arr.append(pml_th)
        pmb_th_arr.append(pmb_th)
        sig_pml_th, sig_pmb_th = sigma_model(l[i], b[i], r[i],
                                             zsun_best, Usun_best, Vsun_best, Wsun_best,
                                             zsun_best_err, Usun_best_err, Vsun_best_err, Wsun_best_err,
                                             V1_best, V2_best, V3_best, V4_best, V5_best, V6_best, V7_best, V8_best,
                                             V9_best, V10_best, V11_best, V1_best_err, V2_best_err, V3_best_err, V4_best_err,
                                             V5_best_err, V6_best_err, V7_best_err, V8_best_err, V9_best_err, V10_best_err, V11_best_err)
        Sig_pml_th.append(sig_pml_th)
        Sig_pmb_th.append(sig_pmb_th)
        delta_pml = (pml[i]-pml_th)/np.sqrt(pml_error[i]**2+sig_pml_th**2)
        delta_pmb = (pmb[i]-pmb_th)/np.sqrt(pmb_error[i]**2+sig_pmb_th**2)
        Delta_pml.append(delta_pml)
        Delta_pmb.append(delta_pmb)
        Delta.append(np.sqrt(delta_pml**2+delta_pmb**2))

    df['pml_th'] = pml_th_arr
    df['pmb_th'] = pmb_th_arr
    df['pml_error_th'] = Sig_pml_th
    df['pmb_error_th'] = Sig_pmb_th
    df['Delta_pml'] = Delta_pml
    df['Delta_pmb'] = Delta_pmb
    df['Delta'] = Delta

    # Store fitted values:
    Zsun_arr[it+1] = zsun_best
    Usun_arr[it+1] = Usun_best
    Vsun_arr[it+1] = Vsun_best
    Wsun_arr[it+1] = Wsun_best
    V1_arr[it+1] = V1_best
    V2_arr[it+1] = V2_best
    V3_arr[it+1] = V3_best
    V4_arr[it+1] = V4_best
    V5_arr[it+1] = V5_best
    V6_arr[it+1] = V6_best
    V7_arr[it+1] = V7_best
    V8_arr[it+1] = V8_best
    V9_arr[it+1] = V9_best
    V10_arr[it+1] = V10_best
    V11_arr[it+1] = V11_best
    Zsun_err_arr[it+1] = zsun_best_err
    Usun_err_arr[it+1] = Usun_best_err
    Vsun_err_arr[it+1] = Vsun_best_err
    Wsun_err_arr[it+1] = Wsun_best_err
    V1_err_arr[it+1] = V1_best_err
    V2_err_arr[it+1] = V2_best_err
    V3_err_arr[it+1] = V3_best_err
    V4_err_arr[it+1] = V4_best_err
    V5_err_arr[it+1] = V5_best_err
    V6_err_arr[it+1] = V6_best_err
    V7_err_arr[it+1] = V7_best_err
    V8_err_arr[it+1] = V8_best_err
    V9_err_arr[it+1] = V9_best_err
    V10_err_arr[it+1] = V10_best_err
    V11_err_arr[it+1] = V11_best_err

    # SIGMA CLIPPING
    indx = np.where(np.array(Delta) < delta_clip)[0]
    num_stars_used.append(len(indx))


# =============================================================================

# RESULTS
# Insert Rotation Curve Patch
V_patch_fitted = np.array([V1_best, V2_best, V3_best, V4_best, V5_best,
                           V6_best, V7_best, V8_best, V9_best, V10_best, V11_best])
V_fit_err = np.array([V1_best_err, V2_best_err, V3_best_err, V4_best_err, V5_best_err,
                      V6_best_err, V7_best_err, V8_best_err, V9_best_err, V10_best_err, V11_best_err])
R_rot, V_rot = patch_array(R_rot_original, V_rot_original, R_patch, V_patch_fitted, R_range_min, R_range_max)

# Oort constants:
dV_dR_1 = ((V2_arr-V1_arr)/(R_patch[1]-R_patch[0]))*1000 # km/s / kpc
dV_dR_2 = ((V3_arr-V2_arr)/(R_patch[2]-R_patch[1]))*1000 # km/s / kpc
dV_dR_3 = ((V4_arr-V3_arr)/(R_patch[3]-R_patch[2]))*1000 # km/s / kpc
dV_dR_4 = ((V5_arr-V4_arr)/(R_patch[4]-R_patch[3]))*1000 # km/s / kpc
dV_dR_5 = ((V6_arr-V5_arr)/(R_patch[5]-R_patch[4]))*1000 # km/s / kpc
dV_dR_6 = ((V7_arr-V6_arr)/(R_patch[6]-R_patch[5]))*1000 # km/s / kpc
dV_dR_7 = ((V8_arr-V7_arr)/(R_patch[7]-R_patch[6]))*1000 # km/s / kpc
dV_dR_8 = ((V9_arr-V8_arr)/(R_patch[8]-R_patch[7]))*1000 # km/s / kpc
dV_dR_9 = ((V10_arr-V9_arr)/(R_patch[9]-R_patch[8]))*1000 # km/s / kpc
dV_dR_10 = ((V11_arr-V10_arr)/(R_patch[10]-R_patch[9]))*1000 # km/s / kpc

dV_dR_1_err = (dV_dR_1/(V2_arr-V1_arr))*np.sqrt(V1_err_arr**2+V2_err_arr**2) # km/s / kpc
dV_dR_2_err = (dV_dR_2/(V3_arr-V2_arr))*np.sqrt(V2_err_arr**2+V3_err_arr**2) # km/s / kpc
dV_dR_3_err = (dV_dR_3/(V4_arr-V3_arr))*np.sqrt(V3_err_arr**2+V4_err_arr**2) # km/s / kpc
dV_dR_4_err = (dV_dR_4/(V5_arr-V4_arr))*np.sqrt(V4_err_arr**2+V5_err_arr**2) # km/s / kpc
dV_dR_5_err = (dV_dR_5/(V6_arr-V5_arr))*np.sqrt(V5_err_arr**2+V6_err_arr**2) # km/s / kpc
dV_dR_6_err = (dV_dR_6/(V7_arr-V6_arr))*np.sqrt(V6_err_arr**2+V7_err_arr**2) # km/s / kpc
dV_dR_7_err = (dV_dR_7/(V8_arr-V7_arr))*np.sqrt(V7_err_arr**2+V8_err_arr**2) # km/s / kpc
dV_dR_8_err = (dV_dR_8/(V9_arr-V8_arr))*np.sqrt(V8_err_arr**2+V9_err_arr**2) # km/s / kpc
dV_dR_9_err = (dV_dR_9/(V10_arr-V9_arr))*np.sqrt(V9_err_arr**2+V10_err_arr**2) # km/s / kpc
dV_dR_10_err = (dV_dR_10/(V11_arr-V10_arr))*np.sqrt(V10_err_arr**2+V11_err_arr**2) # km/s / kpc

dV_dR = np.mean([dV_dR_1[-1], dV_dR_2[-1], dV_dR_3[-1], dV_dR_4[-1], dV_dR_5[-1],
                 dV_dR_6[-1], dV_dR_7[-1], dV_dR_8[-1], dV_dR_9[-1], dV_dR_10[-1]])/(num_rot_curve_params-1)
dV_dR_err = np.sqrt(np.sum(np.array([dV_dR_1_err[-1], dV_dR_2_err[-1], dV_dR_3_err[-1], dV_dR_4_err[-1], dV_dR_5_err[-1],
                                     dV_dR_6_err[-1], dV_dR_7_err[-1], dV_dR_8_err[-1], dV_dR_9_err[-1], dV_dR_10_err[-1]])**2))/(num_rot_curve_params-1)
R0, zsun = 8178, 20
Rc = np.sqrt((R0**2)-(zsun**2))
Vc = VelGal(Rc, R_rot, V_rot, R_range_min, R_range_max)
rr = (Rc-R_patch[0])/(R_patch[-1]-R_patch[0])
Vc_err = np.sqrt((rr*V11_err_arr[-1])**2+((1-rr)*V1_err_arr[-1])**2)
V0_R0 = (Vc/Rc)*1000
A = 0.5*(V0_R0-dV_dR)
B = -0.5*(V0_R0+dV_dR)
AB_err = 0.5*np.sqrt((1000*Vc_err/Rc)**2+(dV_dR_err)**2)
print('\n  Oort Constants:')
print(f'  A = {A:.2f}'+' +/- '+f'{AB_err:.2f} km/s/kpc')
print(f'  B = {B:.2f}'+' +/- '+f'{AB_err:.2f} km/s/kpc\n')

# =============================================================================

# DATA RESULTS
yr2sec = 3600*24*365
pc2km = 3.26156*299792.458*yr2sec
deg2rad = np.pi/180
radsec2masyr = (3600*1000/deg2rad)*yr2sec
RGal, Vr = [], []
for i in range(len(df)):
    rGal, vr, pl, pb = model((l[i], b[i], r[i]), zsun_best, Usun_best, Vsun_best, Wsun_best,
                              V1_best, V2_best, V3_best, V4_best, V5_best, V6_best, V7_best,
                              V8_best, V9_best, V10_best, V11_best, show_more = True)
    RGal.append(np.round(rGal/1000, 2))
    Vr.append(np.round(vr, 1))
df['RGal'] = RGal
df['Vr_th'] = Vr
df['pm_pec'] = np.hypot(df['pml']-df['pml_th'], df['pmb']-df['pmb_th'])
df['V_ang'] = np.round(np.rad2deg(np.arctan2(df['pmb']-df['pmb_th'],df['pml']-df['pml_th']))-90, 1)
df['V_tan'] = np.round(r*df['pm_pec']*(pc2km/radsec2masyr), 1)

df = df.sort_values(by = 'Delta', ascending = False)
df_final = df.reset_index(drop = True)

dd = df[['ID_ALS', 'Name', 'l', 'b', 'RGal', 'pml', 'pml_th', 'pmb', 'pmb_th', 'Vr_th', 'V_ang', 'V_tan', 'Delta']]
dd1 = dd[dd['RGal'].between(5.4, 10.8)]
dd2 = dd[(dd['RGal'].between(5.4, 10.8)) & (dd['V_tan'] > 30)]

# =============================================================================

# HISTOGRAM AND KDE FOR RUNAWAYS
from scipy.stats import gaussian_kde

Vtan = dd1['V_tan'].values
kde = gaussian_kde(Vtan)
kde.set_bandwidth(bw_method = kde.factor*0.3)
x_kde = np.linspace(min(Vtan), max(Vtan), 10000)
y_kde = kde(x_kde)

runaway_threshold = 30 # km/s

plt.style.use('dark_background')
fig = plt.figure(figsize = (10, 5))
bin_size = 2 # km/s
counts, bins, patches = plt.hist(Vtan, bins = np.arange(0, 120, bin_size), color = 'crimson', edgecolor = 'black')
for i in range(len(bins)-1):
    if bins[i] >= runaway_threshold:
        patches[i].set_facecolor('royalblue')
y_kde_normalized = y_kde*len(Vtan)*bin_size
plt.plot(x_kde, y_kde_normalized, color = 'white', linewidth = 2.5, alpha = 0.5)
plt.axvline(x = runaway_threshold, linestyle = '--', color = 'gold', linewidth = 2.5)
plt.text(runaway_threshold+1, 77, s = r'Runaway threshold $= 30$ km/s', color = 'gold', fontsize = 10,
         horizontalalignment = 'left', verticalalignment = 'center', rotation = 90)
plt.text(runaway_threshold-2, 44, s = fr'${len(dd1)-len(dd2)}$', color = 'crimson', fontsize = 21,
         horizontalalignment = 'right', verticalalignment = 'center')
plt.text(runaway_threshold+2, 44, s = fr'${len(dd2)}$', color = 'royalblue', fontsize = 21,
         horizontalalignment = 'left', verticalalignment = 'center')
plt.text(runaway_threshold-2, 36, s = fr'${100*(len(dd1)-len(dd2))/len(dd1):.1f} \%$', color = 'crimson', fontsize = 16,
         horizontalalignment = 'right', verticalalignment = 'center')
plt.text(runaway_threshold+2, 36, s = fr'${100*len(dd2)/len(dd1):.1f} \%$', color = 'royalblue', fontsize = 16,
         horizontalalignment = 'left', verticalalignment = 'center')
plt.minorticks_on()
plt.grid(which = 'major', linestyle = ':', alpha = 0.35)
plt.grid(which = 'minor', linestyle = ':', alpha = 0.20)
plt.tick_params(axis = 'both', direction = 'in', which = 'major', length = 5, width = 0.7, labelsize = 12)
plt.tick_params(axis = 'both', direction = 'in', which = 'minor', length = 3, width = 0.7)
plt.tick_params(axis = 'both', which = 'major', bottom = True, top = True, left = True, right = True, labeltop = True, labelbottom = True)
plt.tick_params(axis = 'both', which = 'minor', bottom = True, top = True, left = True, right = True)
plt.xlim(0, 100)
plt.ylim(0, np.max(counts)*1.05)
plt.xlabel(r'$V_{\perp}$ [km/s]', fontsize = 13)
plt.ylabel('Number of stars', fontsize = 13)
custom_ticks = np.arange(0, 100+5, 5)
custom_labels = [str(i) for i in custom_ticks]
plt.xticks(custom_ticks)
plt.savefig('Runaway Histogram 11.png', format = 'png', dpi = 400)
plt.show()

# =============================================================================

# PLOT ROTATION CURVE
import matplotlib.gridspec as gridspec

# KDE
RGal = np.array(RGal)*1000
kde = gaussian_kde(RGal)
kde.set_bandwidth(bw_method = kde.factor*0.25)
x_kde = np.linspace(min(RGal), max(RGal), 1000)
y_kde = kde(x_kde)

# PLOT
fig = plt.figure(figsize = (12, 5))
gs = gridspec.GridSpec(2, 1, height_ratios = [1.5, 3])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex = ax1)

bin_size = 200 # pc
bins = np.arange(min(RGal), max(RGal)+bin_size, bin_size)
counts, bin_edges, p = ax1.hist(RGal, bins = bins, color = 'grey', alpha = 0.60)
ax1.hist(df[df.index.isin(indx)]['RGal']*1000, bins = bins, color = 'white', alpha = 0.60, edgecolor = 'black')
y_kde_normalized = y_kde*len(RGal)*bin_size
ax1.plot(x_kde, y_kde_normalized, color = 'crimson', linewidth = 2.5)
ax1.fill_between(x_kde, y_kde_normalized, color = 'crimson', alpha = 0.2)
ax1.axvline(x = R_range_min, linestyle = '--', color = 'white', alpha = 0.4, zorder = 0)
ax1.axvline(x = R_range_max, linestyle = '--', color = 'white', alpha = 0.4, zorder = 0)
ax1.text(1250, 75, s = fr'${num_stars_used[-1]}$ stars used for the fit', color = 'white', fontsize = 13,
         horizontalalignment = 'left', verticalalignment = 'center')
ax1.minorticks_on()
ax1.grid(which = 'major', linestyle = ':', alpha = 0.35)
ax1.grid(which = 'minor', linestyle = ':', alpha = 0.20)
ax1.tick_params(axis = 'both', direction = 'in', which = 'major', length = 5, width = 0.7, labelsize = 12)
ax1.tick_params(axis = 'both', direction = 'in', which = 'minor', length = 3, width = 0.7)
ax1.tick_params(axis = 'both', which = 'major', bottom = True, top = True, left = True, right = True, labeltop = True, labelbottom = False)
ax1.tick_params(axis = 'both', which = 'minor', bottom = True, top = True, left = True, right = True)
ax1.set_xlim(0, 13000)
ax1.set_ylim(0, np.max(counts)*1.15)
custom_ticks = np.arange(0, 13000+1, 1000)
custom_labels = [str(int(i/1000)) for i in custom_ticks]
ax1.set_xticks(custom_ticks)
ax1.set_xticklabels(custom_labels)
ax1.set_ylabel('Number of stars', fontsize = 13)

x = np.linspace(0, 15000, 1000)
ax2.plot(x, VelGal(x, R_rot_original, V_rot_original, R_range_min, R_range_max), '-c', linewidth = 2.5, zorder = 1)
ax2.plot(R_patch, V_patch_fitted, linestyle = '-', color = 'crimson', linewidth = 2, zorder = 2)
ax2.scatter(Rc, Vc, c = 'yellow', s = 70, zorder = 3)
ax2.text(Rc, Vc+5, 'Sun', horizontalalignment = 'center', verticalalignment = 'bottom', color = 'yellow', fontsize = 14)
ax2.plot([Rc, Rc], [Vc, 0], linestyle = '--', color = 'yellow')
ax2.text(1250, 255, fr'$V_{{c}} = {round_to_significant_digits(Vc, Vc_err)[0]} \pm {round_to_significant_digits(Vc, Vc_err)[1]}$ km/s',
         horizontalalignment = 'left', verticalalignment = 'center', color = 'yellow', fontsize = 13)
ax2.text(1250, 246, fr'$U_{{\bigodot}} = {round_to_significant_digits(Usun_best, Usun_best_err)[0]} \pm {round_to_significant_digits(Usun_best, Usun_best_err)[1]}$ km/s',
         horizontalalignment = 'left', verticalalignment = 'center', color = 'yellow', fontsize = 13)
ax2.text(1250, 237, fr'$V_{{\bigodot}} = {round_to_significant_digits(Vsun_best, Vsun_best_err)[0]} \pm {round_to_significant_digits(Vsun_best, Vsun_best_err)[1]}$ km/s',
         horizontalalignment = 'left', verticalalignment = 'center', color = 'yellow', fontsize = 13)
ax2.text(1250, 228, fr'$W_{{\bigodot}} = {round_to_significant_digits(Wsun_best, Wsun_best_err)[0]} \pm {round_to_significant_digits(Wsun_best, Wsun_best_err)[1]}$ km/s',
         horizontalalignment = 'left', verticalalignment = 'center', color = 'yellow', fontsize = 13)
ax2.text(1250, 195, fr'$A = {round_to_significant_digits(A, AB_err)[0]} \pm {round_to_significant_digits(A, AB_err)[1]}$ km/s/kpc',
         horizontalalignment = 'left', verticalalignment = 'center', color = 'cyan', fontsize = 13)
ax2.text(1250, 186, fr'$B = {round_to_significant_digits(B, AB_err)[0]} \pm {round_to_significant_digits(B, AB_err)[1]}$ km/s/kpc',
         horizontalalignment = 'left', verticalalignment = 'center', color = 'cyan', fontsize = 13)
for i in range(len(R_patch)):
    R_point, V_point, V_point_err = R_patch[i], V_patch_fitted[i], V_fit_err[i]
    V_point_err_low, V_point_err_high = V_point-V_point_err, V_point+V_point_err
    ax2.plot([R_point, R_point], [V_point, 0], linestyle = ':', color = 'white')
    ax2.plot([R_point, R_point], [V_point_err_low, V_point_err_high], linestyle = '-', linewidth = 2, color = 'crimson')
    ax2.plot([R_point-100, R_point+100], [V_point_err_low, V_point_err_low], linestyle = '-', linewidth = 2, color = 'crimson')
    ax2.plot([R_point-100, R_point+100], [V_point_err_high, V_point_err_high], linestyle = '-', linewidth = 2, color = 'crimson')
    ax2.scatter(R_point, V_point, c = 'crimson', s = 50)

ax2.minorticks_on()
ax2.grid(which = 'major', linestyle = ':', alpha = 0.35)
ax2.grid(which = 'minor', linestyle = ':', alpha = 0.20)
ax2.tick_params(axis = 'both', direction = 'in', which = 'major', length = 5, width = 0.7, labelsize = 12)
ax2.tick_params(axis = 'both', direction = 'in', which = 'minor', length = 3, width = 0.7)
ax2.tick_params(axis = 'both', which = 'major', bottom = True, top = True, left = True, right = True, labelright = True)
ax2.tick_params(axis = 'both', which = 'minor', bottom = True, top = True, left = True, right = True)
ax2.set_xlim(0, 13000)
ax2.set_ylim(175, 265)
ax2.set_xlabel(r'$R$ [kpc]', fontsize = 14)
ax2.set_ylabel(r'$V$ [km/s]', fontsize = 14)

plt.subplots_adjust(hspace = 0)
plt.savefig('Rotation Curve 11.png', format = 'png', dpi = 400)
plt.show()
