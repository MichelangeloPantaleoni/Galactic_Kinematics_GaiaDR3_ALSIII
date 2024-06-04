import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import cmasher as cmr
from scipy import interpolate
import matplotlib.pyplot as plt
from astropy.table import Table
from astroquery.gaia import Gaia
plt.rcParams["text.usetex"] = True

def dataframe_to_fits(df, name):
    t = Table.from_pandas(df)
    t.write(name+'.fits', format = 'fits', overwrite = True)

def interpol(x, x_data, y_data):
    n = len(x)
    if isinstance(x_data, list):
        x_data = np.array(x_data)
    if isinstance(y_data, list):
        y_data = np.array(y_data)
    y = np.zeros(n)
    for i in range(n):
        if sum(x_data > x[i]) == len(x_data):
            # If x is below all the x_data points, then retrieve the lower margin of the data
            y[i] = y_data[0]
        elif sum(x_data > x[i]) == 0:
            # If x is above all the x_data points, then retrieve the upper margin of the data
            y[i] = y_data[-1]
        elif sum(x_data == x[i]) == 0:
            # If x is inside the range covered by x_data and does not coincide with any x_data point,
            # then retrieve the linearly interpolated value between the closest points.
            x1 = np.max(x_data[x_data < x[i]])
            x2 = np.min(x_data[x_data > x[i]])
            y1 = y_data[x_data == x1]
            y2 = y_data[x_data == x2]
            y[i] = (((x[i]-x1)*(y2-y1)/(x2-x1))+y1)[0]
        else:
            # If x coincides with an x_data point then use the correspondent y_data value
            y[i] = y_data[x[i] == x_data][0]
    return y

def parallax_correct(df, mode = 'Apellaniz'):
    gmag, lat, npar = df[['phot_g_mean_mag', 'ecl_lat', 'astrometric_params_solved']].values.T
    gmag[np.where(np.isnan(gmag))[0]] = 21.0 # Stars with non-valid G are assumed to be faint
    nueff = (df['nu_eff_used_in_astrometry'].fillna(0)+df['pseudocolour'].fillna(0)).values

    # For objects with 6-parameter solutions (Lindegren and Apellaniz do the same)
    gcut6 = [  6.0 ,  10.8 ,  11.2 ,  11.8 ,  12.2 ,  12.9 ,  13.1 ,  15.9 ,  16.1 ,  17.5 ,  19.0 ,  20.0 ,  21.0 ]
    q600  = [-27.85, -28.91, -26.72, -29.04, -12.39, -18.99, -38.29, -36.83, -28.37, -24.68, -15.32, -13.73, -29.53]
    q601  = [ -7.78,  -3.57,  -8.74,  -9.69,  -2.16,  -1.93,  +2.59,  +4.20,  +1.99,  -1.37,  +4.01, -10.92, -20.34]
    q602  = [+27.47, +22.92,  +9.36, +13.63, +10.23, +15.90, +16.20, +15.76,  +9.28,  +3.52,  -6.03,  -8.30, -18.74]
    q610  = [-32.1 ,  +7.7 , -30.3 , -49.4 , -92.6 , -57.2 , -10.5 , +22.3 , +50.4 , +86.8 , +29.2 , -74.4 , -39.5 ]
    q611  = [+14.4 , +12.6 ,  +5.6 , +36.3 , +19.8 ,  -8.0 ,  +1.4 , +11.1 , +17.2 , +19.8 , +14.1 ,+196.4 ,+326.8 ]
    q612  = [ +9.5 ,  +1.6 , +17.2 , +17.7 , +27.6 , +19.9 ,  +0.4 , +10.0 , +13.7 , +21.3 ,  +0.4 , -42.0 ,-262.3 ]
    q620  = [  -67.,  -572., -1104., -1129.,  -365.,  -554.,  -960., -1367., -1351., -1380.,  -563.,  +536., +1598.]

    # For objects with 5-parameter solutions:
    if mode == 'Apellaniz':
        # Maíz Apellaniz 2022 [2022A%26A...657A.130M]
        gcut5 = [  6.0 ,   7.4 ,   9.2 ,  10.8 ,  11.2 ,  11.8 ,  12.2 ,  12.9 ,  13.1 ,  15.9 ,  16.1 ,  17.5 ,  18.0 ,  19.0 ,  20.0 ,  21.0 ]
        q500  = [-54.33,  -8.17, -27.11, -20.03, -33.38, -36.95, -16.99, -26.49, -37.51, -32.82, -33.18, -25.02, -22.88, -18.40, -12.65, -18.22]
        q501  = [-11.97, -10.06,  -7.60,  -2.78, -12.21, -11.55,  -3.67, -10.63,  +3.33,  +5.15,  -1.31,  +5.83,   0.40,  +5.98,  -4.57, -15.24]
        q502  = [+25.39, +24.12, +22.48, +10.48,  +5.51,  -1.65, +15.81, +21.77, +20.41,  +6.50,  +5.48,  +6.57,  -5.10,  -6.46,  -7.46, -18.54]
        q510  = [-31.1 , -13.4 ,  +9.3 , +11.6 ,-132.3 ,-158.7 ,-109.9 , -76.0 ,  -2.9 ,  -9.10, -56.8 , -39.2 , -46.5 ,   0   ,   0   ,   0   ]
        q511  = [+19.1 , +23.7 , +29.6 , +34.8 , -10.2 , +13.2 , +63.2 , +43.2 , +29.6 , +12.2 , -38.1 , -29.1 , -35.4 ,  +5.5 , +97.9 ,+128.2 ]
        q520  = [-2529., -2529., -2529., -2529., -2529., -2529., -3625., -4353., -1675., -1341., -1705., -1284.,  -896.,   0   ,   0   ,   0   ]
        q530  = [  0   ,   0   ,   0   ,   0   ,   0   ,   0   ,   0   ,   0   , +32.1 ,+168.0 ,+112.1 ,+196.3 ,+126.5 ,   0   ,   0   ,   0   ]
        q540  = [-358.1,-358.1 ,-358.1 , -78.6 ,+203.1 ,-155.3 ,-144.2 , +23.6 , +99.5 ,+129.3 ,+153.1 ,+218.0 ,+190.2 ,+276.6 ,   0   ,   0   ]
    elif mode == 'Lindegren':
        # Lindegren et al. 2021 [2021A%26A...649A...4L]
        gcut5 = [  6.0 ,  10.8 ,  11.2 ,   11.8 ,   12.2 ,  12.9 ,   13.1 ,   15.9 ,   16.1 ,   17.5 ,   19.0 ,  20.0 ,   21.0 ]
        q500  = [-26.98, -27.23, -30.33,  -33.54,  -13.65, -19.53,  -37.99,  -38.33,  -31.05,  -29.18,  -18.40, -12.65,  -18.22]
        q501  = [ -9.62,  -3.07,  -9.23,  -10.08,   -0.07,  -1.64,   +2.63,   +5.61,   +2.83,   -0.09,   +5.98,  -4.57,  -15.24]
        q502  = [+27.40, +23.04,  +9.08,  +13.28,   +9.35, +15.86,  +16.14,  +15.42,   +8.59,   +2.41,   -6.46,  -7.46,  -18.54]
        q510  = [-25.1 , +35.3 , -88.4 , -126.7 , -111.4 , -66.8 ,   -5.7 ,    0   ,    0   ,    0   ,    0   ,   0   ,    0   ]
        q511  = [ -0.0 , +15.7 , -11.8 ,  +11.6 ,  +40.6 , +20.6 ,  +14.0 ,  +18.7 ,  +15.5 ,  +24.5 ,   +5.5 , +97.9 , +128.2 ]
        q520  = [-1257., -1257., -1257.,  -1257.,  -1257., -1257.,  -1257.,  -1189.,  -1404.,  -1165.,    0   ,   0   ,    0   ]
        q530  = [  0   ,   0   ,   0   ,    0   ,    0   ,   0   , +107.9 , +243.8 , +105.5 , +189.7 ,    0   ,   0   ,    0   ]
        q540  = [  0   ,   0   ,   0   ,    0   ,    0   ,   0   , +104.3 , +155.2 , +170.7 , +325.0 , +276.6 ,   0   ,    0   ]
    else:
        raise ValueError("  Wrong 'mode': You have to choose either 'Apellaniz' or 'Lindegren'.  ")

    c1 = -0.24*(nueff <= 1.24).astype(int)+(nueff-1.48)*((nueff > 1.24) & (nueff <= 1.72)).astype(int)+0.24*(nueff > 1.72).astype(int)
    c2 = +0.24**3*(nueff <= 1.24).astype(int)+(1.48-nueff)**3*((nueff > 1.24) & (nueff <= 1.48)).astype(int)
    c3 = (nueff-1.24)*(nueff <= 1.24).astype(int)
    c4 = (nueff-1.72)*(nueff > 1.72).astype(int)
    b1 = np.sin(np.deg2rad(lat))
    b2 = np.sin(np.deg2rad(lat))*np.sin(np.deg2rad(lat))-1/3
    qq500 = interpol(gmag, gcut5, q500)
    qq501 = interpol(gmag, gcut5, q501)
    qq502 = interpol(gmag, gcut5, q502)
    qq510 = interpol(gmag, gcut5, q510)
    qq511 = interpol(gmag, gcut5, q511)
    qq520 = interpol(gmag, gcut5, q520)
    qq530 = interpol(gmag, gcut5, q530)
    qq540 = interpol(gmag, gcut5, q540)
    qq600 = interpol(gmag, gcut6, q600)
    qq601 = interpol(gmag, gcut6, q601)
    qq602 = interpol(gmag, gcut6, q602)
    qq610 = interpol(gmag, gcut6, q610)
    qq611 = interpol(gmag, gcut6, q611)
    qq612 = interpol(gmag, gcut6, q612)
    qq620 = interpol(gmag, gcut6, q620)
    z5 = qq500 + qq501*b1 + qq502*b2 + qq510*c1 + qq511*c1*b1 + qq520*c2 + qq530*c3 + qq540*c4
    z6 = qq600 + qq601*b1 + qq602*b2 + qq610*c1 + qq611*c1*b1 + qq612*c1*b2 + qq620*c2
    zpvals = z5*(npar == 31).astype(int)+z6*(npar == 95).astype(int)+999999*(npar == 3).astype(int) # Returns 999999 for npar = 3
    df['Plx_zero'] = zpvals/1000 # Change from microarcsec to milliarcsec
    df['Plx_zero'] = df['Plx_zero'].replace(999999/1000, np.nan) # No parallax <=> npar = 3 => NaNs
    df['Plx_corr'] = df['parallax']-df['Plx_zero'] # Correct parallax with the zero point bias
    return df

def parallax_error_correct(df):
    # Maíz Apellaniz 2022 [2022A%26A...657A.130M], Appendix A
    plx_err, gmag, ruwe, npar = df[['parallax_error', 'phot_g_mean_mag', 'ruwe', 'astrometric_params_solved']].values.T
    gmag[np.where(np.isnan(gmag))[0]] = 21.0 # Stars with non-valid G are assumed to be faint
    gref = np.array([ 6.50,  7.50,  8.50,  9.50, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75,
                     13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75])
    kref = np.array([2.62, 2.38, 2.06, 1.66, 1.22, 1.41, 1.76, 1.76, 1.90, 1.92,
                     1.61, 1.50, 1.39, 1.35, 1.24, 1.20, 1.19, 1.18, 1.18, 1.14])
    reduced_gmag = (gmag < 6)*6+(gmag > 18)*18+((gmag <= 18) & (gmag >= 6))*gmag
    tck = interpolate.splrep(gref, kref, k = 1)
    k = interpolate.splev(reduced_gmag, tck, der = 0)
    geref = [6.00, 12.50, 13.50, 14.50, 15.50, 16.50, 17.50]
    keref = [0.50,  0.50,  1.01,  1.28,  1.38,  1.44,  1.32]
    new_tck = interpolate.splrep(geref, keref, k = 1)
    new_k = interpolate.splev(reduced_gmag, new_tck, der = 0)
    k = k*(1+((new_k <= 0.5)*0.5+(new_k > 0.5)*new_k)*(ruwe > 1.4))
    k = k*(1+(npar == 95*np.ones(len(npar)))*0.25)
    df['Plx_err_corr'] = np.sqrt((k*plx_err)**2+0.0103**2)
    return df

def add_gaia_data(df, gaia_columns, gaia_id_col_name = 'ID_DR3', max_chunk_size = 10000):
    gaia_ids_all = df[(df[gaia_id_col_name] != '') & (~df[gaia_id_col_name].isna())][gaia_id_col_name].values
    num_chunks = int(len(gaia_ids_all)/max_chunk_size)+(1 if len(gaia_ids_all)%max_chunk_size != 0 else 0)
    for i in range(num_chunks):
        if num_chunks > 1:
            print(f'  Querying Gaia DR3 data in {num_chunks} chunks of size {max_chunk_size}')
            gaia_ids = gaia_ids_all[i*max_chunk_size:(i+1)*max_chunk_size]
        else:
            print('  Querying Gaia DR3')
            gaia_ids = gaia_ids_all
        df_chunk = df[df[gaia_id_col_name].isin(gaia_ids)]
        if len(gaia_ids) == 1:
            adql_command_gaia = "SELECT source_id, "+', '.join(gaia_columns)+ " \
                                 FROM gaiadr3.gaia_source \
                                 WHERE source_id = '" + \
                                 str(gaia_ids[0])+"'"
        else:
            adql_command_gaia = "SELECT source_id, "+', '.join(gaia_columns)+ " \
                                 FROM gaiadr3.gaia_source \
                                 WHERE source_id IN " + \
                                 str(tuple(gaia_ids))
        job_gaia = Gaia.launch_job_async(adql_command_gaia, dump_to_file = False)
        df_job_gaia = job_gaia.get_results().to_pandas()
        df_job_gaia['SOURCE_ID'] = df_job_gaia['SOURCE_ID'].astype(str)
        df_chunk = pd.merge(left = df_chunk, right = df_job_gaia, left_on = gaia_id_col_name, right_on = 'SOURCE_ID', how = 'left')
        df_chunk.drop(['SOURCE_ID'], axis = 1, inplace = True)
        if i == 0:
            df_final = df_chunk.copy()
        else:
            df_final = pd.concat([df_final, df_chunk])
    df_final = df_final.reset_index(drop = True)
    return df_final

def correct_proper_motion(pmra, pmdec, ra, dec, phot_g_mean_mag):
    if np.isnan(phot_g_mean_mag) or (phot_g_mean_mag >= 13.0):
        return pmra, pmdec
    else:
        table = [[  0.00,   9.00,   9.50,  10.00, 10.50, 11.00, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75],
                 [  9.00,   9.50,  10.00,  10.50, 11.00, 11.50, 11.75, 12.00, 12.25, 12.50, 12.75, 13.00],
                 [ 18.40,  14.00,  12.80,  13.60, 16.20, 19.40, 21.80, 17.70, 21.30, 25.70, 27.30, 34.90],
                 [ 33.80,  30.70,  31.40,  35.70, 50.00, 59.90, 64.20, 65.60, 74.80, 73.60, 76.60, 68.90],
                 [-11.30, -19.40, -11.80, -10.50,  2.10,  0.20,  1.00, -1.90,  2.10,  1.00,  0.50, -2.90]]
        table = np.array(table)
        Gmin = table[0]
        Gmax = table[1]
        omegaX = table[2][(Gmin <= phot_g_mean_mag) & (Gmax > phot_g_mean_mag)][0]
        omegaY = table[3][(Gmin <= phot_g_mean_mag) & (Gmax > phot_g_mean_mag)][0]
        omegaZ = table[4][(Gmin <= phot_g_mean_mag) & (Gmax > phot_g_mean_mag)][0]
        sin_ra,  cos_ra  = np.sin(np.deg2rad(ra)),  np.cos(np.deg2rad(ra))
        sin_dec, cos_dec = np.sin(np.deg2rad(dec)), np.cos(np.deg2rad(dec))
        pmra_corr = -sin_dec*cos_ra*omegaX-sin_dec*sin_ra*omegaY+cos_dec*omegaZ
        pmdec_corr = sin_ra*omegaX-cos_ra*omegaY
        return pmra-pmra_corr/1000., pmdec-pmdec_corr/1000.

def correct_proper_motion_error(df, only_internal_uncertainty = False):
    # The same k as in the parallax correction (Maíz Apellaniz 2022
    # [2022A%26A...657A.130M], Appendix A), but with a systematic uncertainty
    # of 23 µas/a, from Lindegren et al. 2021 [].
    pmra_err, pmdec_err, gmag, ruwe, npar = df[['pmra_error', 'pmdec_error', 'phot_g_mean_mag', 'ruwe', 'astrometric_params_solved']].values.T
    gmag[np.where(np.isnan(gmag))[0]] = 21.0 # Stars with non-valid G are assumed to be faint
    gref = np.array([ 6.50,  7.50,  8.50,  9.50, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75,
                     13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75])
    kref = np.array([2.62, 2.38, 2.06, 1.66, 1.22, 1.41, 1.76, 1.76, 1.90, 1.92,
                     1.61, 1.50, 1.39, 1.35, 1.24, 1.20, 1.19, 1.18, 1.18, 1.14])
    reduced_gmag = (gmag < 6)*6+(gmag > 18)*18+((gmag <= 18) & (gmag >= 6))*gmag
    tck = interpolate.splrep(gref, kref, k = 1)
    k = interpolate.splev(reduced_gmag, tck, der = 0)
    geref = [6.00, 12.50, 13.50, 14.50, 15.50, 16.50, 17.50]
    keref = [0.50,  0.50,  1.01,  1.28,  1.38,  1.44,  1.32]
    new_tck = interpolate.splrep(geref, keref, k = 1)
    new_k = interpolate.splev(reduced_gmag, new_tck, der = 0)
    k = k*(1+((new_k <= 0.5)*0.5+(new_k > 0.5)*new_k)*(ruwe > 1.4))
    k = k*(1+(npar == 95*np.ones(len(npar)))*0.25)
    if only_internal_uncertainty:
        df['pmra_error_corrected'] = k*pmra_err
        df['pmdec_error_corrected'] = k*pmdec_err
    else:
        df['pmra_error_corrected'] = np.sqrt((k*pmra_err)**2+0.0230**2)
        df['pmdec_error_corrected'] = np.sqrt((k*pmdec_err)**2+0.0230**2)
    return df

def galactic_proper_motion_full(ra, dec, ra_error, dec_error, parallax, parallax_error, pmra, pmdec, pmra_error, pmdec_error,
                                ra_dec_corr = 0, ra_parallax_corr = 0, ra_pmra_corr = 0, ra_pmdec_corr = 0, dec_parallax_corr = 0,
                                dec_pmra_corr = 0, dec_pmdec_corr = 0, parallax_pmra_corr = 0, parallax_pmdec_corr = 0, pmra_pmdec_corr = 0):
    # Gaia DR1 Documentation or Gaia DR2 Documentation, section 3.1.7 Transformations of astrometric data and error propagation:
    # https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html#Ch3.E57

    # ICRS coordinates of the north Galactic pole and the longitude of intersecting planes
    ra_pole, dec_pole, l_omega = np.deg2rad(192.85948), np.deg2rad(27.12825), np.deg2rad(32.93192)

    # Rotation matrices
    Rz_lomg = np.array([[ np.cos(-l_omega),  np.sin(-l_omega), 0],
                        [-np.sin(-l_omega),  np.cos(-l_omega), 0],
                        [                0,                 0, 1]])
    Rx_90dec = np.array([[1,                         0,                         0],
                         [0,  np.cos(np.pi/2-dec_pole),  np.sin(np.pi/2-dec_pole)],
                         [0, -np.sin(np.pi/2-dec_pole),  np.cos(np.pi/2-dec_pole)]])
    Rz_a90 = np.array([[ np.cos(ra_pole+np.pi/2),  np.sin(ra_pole+np.pi/2), 0],
                       [-np.sin(ra_pole+np.pi/2),  np.cos(ra_pole+np.pi/2), 0],
                       [                       0,                        0, 1]])
    AgT = Rz_lomg @ Rx_90dec @ Rz_a90 # Eq. 3.60

    # Galactic coordinates
    ra_rad, dec_rad = np.deg2rad(ra), np.deg2rad(dec)
    r_icrs = np.array([np.cos(ra_rad)*np.cos(dec_rad), np.sin(ra_rad)*np.cos(dec_rad), np.sin(dec_rad)]) # Eq. 3.57
    r_gal = AgT.dot(r_icrs) # Eq. 3.59
    l_rad = np.arctan2(r_gal[1], r_gal[0]) # Eq. 3.63
    b_rad = np.arctan2(r_gal[2], np.hypot(r_gal[1], r_gal[0])) # Eq. 3.63
    l = (np.rad2deg(l_rad)+360)%360 # Between 0º and 360º
    b = np.rad2deg(b_rad) # Between -90º and 90º

    # Galactic proper motion
    p_icrs = np.array([-np.sin(ra_rad), np.cos(ra_rad), 0]) # Eq. 3.64
    q_icrs = np.array([-np.cos(ra_rad)*np.sin(dec_rad), -np.sin(ra_rad)*np.sin(dec_rad), np.cos(dec_rad)]) # Eq. 3.64
    pm_icrs = p_icrs*pmra+q_icrs*pmdec # Eq. 3.66
    p_gal = np.array([-np.sin(l_rad), np.cos(l_rad), 0]) # Eq. 3.65
    q_gal = np.array([-np.cos(l_rad)*np.sin(b_rad), -np.sin(l_rad)*np.sin(b_rad), np.cos(b_rad)]) # Eq. 3.65
    pm_gal = AgT.dot(pm_icrs) # Eq. 3.68
    pml = p_gal.dot(pm_gal) # Eq. 3.70
    pmb = q_gal.dot(pm_gal) # Eq. 3.70

    # Transform the covariance matrix
    ra_dec_cov = ra_error*dec_error*ra_dec_corr
    ra_parallax_cov = ra_error*parallax_error*ra_parallax_corr
    ra_pmra_cov = ra_error*pmra_error*ra_pmra_corr
    ra_pmdec_cov = ra_error*pmdec_error*ra_pmdec_corr
    dec_parallax_cov = dec_error*parallax_error*dec_parallax_corr
    dec_pmra_cov = dec_error*pmra_error*dec_pmra_corr
    dec_pmdec_cov = dec_error*pmdec_error*dec_pmdec_corr
    parallax_pmra_cov = parallax_error*pmra_error*parallax_pmra_corr
    parallax_pmdec_cov = parallax_error*pmdec_error*parallax_pmdec_corr
    pmra_pmdec_cov = pmra_error*pmdec_error*pmra_pmdec_corr
    C = np.array([[ra_error**2    , ra_dec_cov      , ra_parallax_cov   , ra_pmra_cov      , ra_pmdec_cov      ],
                  [ra_dec_cov     , dec_error**2    , dec_parallax_cov  , dec_pmra_cov     , dec_pmdec_cov     ],
                  [ra_parallax_cov, dec_parallax_cov, parallax_error**2 , parallax_pmra_cov, parallax_pmdec_cov],
                  [ra_pmra_cov    , dec_pmra_cov    , parallax_pmra_cov , pmra_error**2    , pmra_pmdec_cov    ],
                  [ra_pmdec_cov   , dec_pmdec_cov   , parallax_pmdec_cov, pmra_pmdec_cov   , pmdec_error**2    ]]) # Eq. 3.74
    G = np.array([p_gal, q_gal]).dot(AgT.dot(np.array([p_icrs, q_icrs]).T)) # Eq. 3.80
    J = np.array([[G[0][0], G[0][1], 0,       0,       0],
                  [G[1][0], G[1][1], 0,       0,       0],
                  [      0,       0, 1,       0,       0],
                  [      0,       0, 0, G[0][0], G[0][1]],
                  [      0,       0, 0, G[1][0], G[1][1]]]) # Eq. 3.79
    C_gal = J @ C @ J.T # Eq. 3.78

    # Transformed uncertainties and correlations
    l_error, b_error, parallax_error, pml_error, pmb_error = np.sqrt(np.diag(C_gal))
    l_b_corr = C_gal[0][1]/(l_error*b_error)
    l_parallax_corr = C_gal[0][2]/(l_error*parallax_error)
    l_pml_corr = C_gal[0][3]/(l_error*pml_error)
    l_pmb_corr = C_gal[0][4]/(l_error*pmb_error)
    b_parallax_corr = C_gal[1][2]/(b_error*parallax_error)
    b_pml_corr = C_gal[1][3]/(b_error*pml_error)
    b_pmb_corr = C_gal[1][4]/(b_error*pmb_error)
    parallax_pml_corr = C_gal[2][3]/(parallax_error*pml_error)
    parallax_pmb_corr = C_gal[2][4]/(parallax_error*pmb_error)
    pml_pmb_corr = C_gal[3][4]/(pml_error*pmb_error)

    # Pack results together
    vals = [l, b, parallax, pml, pmb]
    errors = [l_error, b_error, parallax_error, pml_error, pmb_error]
    corr = [l_b_corr, l_parallax_corr, l_pml_corr, l_pmb_corr, b_parallax_corr,
            b_pml_corr, b_pmb_corr, parallax_pml_corr, parallax_pmb_corr, pml_pmb_corr]
    return [vals, errors, corr]

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
    V = VelGal(R, df_rotcurve['R'].values, df_rotcurve['V'].values) # km/s
    # V = Vc
    omega = V/R

    Vr = -Usun*cb*cl+(omega*Rc-Vsun-Vc)*cb*sl-Wsun*sb # km/s
    pml = (1/r)*(Usun*sl+(omega*Rc-Vsun-Vc)*cl-omega*r*(ss*cb+s*sb*cl))*(radsec2masyr/pc2km) # mas/yr
    pmb = (1/r)*(Usun*sb*cl+(Vsun+Vc-omega*Rc)*sb*sl-Wsun*cb+omega*s*r*sl)*(radsec2masyr/pc2km) # mas/yr
    return Vr, pml, pmb

# LOAD ROTATION CURVE
file_path = 'MW Grand Rotation Curve 2021.dat'
df_rotcurve = pd.read_csv(file_path, header = 0, delimiter = r'\s+')
df_rotcurve = df_rotcurve.rename(columns = {'R(kpc),':'R', 'dR(St.dev),':'dR', 'V(km/s),':'V', 'dV(St.dev)':'dV'})
df_rotcurve['R'] = df_rotcurve['R']*1000

# LOAD ALS
df = pd.read_pickle('ALS.pkl')
for col in ['Dist_P16', 'Dist_P50', 'Dist_P84']:
    df[col] = df[col].astype(float)

# DEFINE SAMPLE
df = df[df['Cat'].isin(['Ma', 'S05'])]
df = df[~df['Dist_P50'].isna()]
df = df.reset_index(drop = True)

# DOWNLOAD GAIA PROPER MOTION DATA
cols = ['ra_error', 'dec_error', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error',
        'ra_dec_corr', 'ra_parallax_corr', 'ra_pmra_corr', 'ra_pmdec_corr',
        'dec_parallax_corr', 'dec_pmra_corr', 'dec_pmdec_corr',
        'parallax_pmra_corr', 'parallax_pmdec_corr', 'pmra_pmdec_corr']
df = add_gaia_data(df, gaia_columns = cols)
# Extract data for the next loops
ra, dec, phot_g_mean_mag = df['ra'].values, df['dec'].values, df['phot_g_mean_mag'].values
ra_error, dec_error = df['ra_error'].values, df['dec_error'].values
parallax, parallax_error = df['parallax'].values, df['parallax_error'].values
pmra, pmdec = df['pmra'].values, df['pmdec'].values
pmra_error, pmdec_error = df['pmra_error'].values, df['pmdec_error'].values
ra_dec_corr, ra_parallax_corr = df['ra_dec_corr'].values, df['ra_parallax_corr'].values
ra_pmra_corr, ra_pmdec_corr = df['ra_pmra_corr'].values, df['ra_pmdec_corr'].values
dec_parallax_corr, dec_pmra_corr = df['dec_parallax_corr'].values, df['dec_pmra_corr'].values
dec_pmdec_corr, parallax_pmra_corr = df['dec_pmdec_corr'].values, df['parallax_pmra_corr'].values
parallax_pmdec_corr, pmra_pmdec_corr = df['parallax_pmdec_corr'].values, df['pmra_pmdec_corr'].values

# CORRECT PARALLAX
print('\n  Correcting parallaxes')
df = parallax_correct(df)

# CORRECT PARALLAX UNCERTAINTIES
print('  Correcting parallax uncertainties')
df = parallax_error_correct(df)
df = df[~df['parallax'].isna()]
df = df.reset_index(drop = True)

# PARALLAX QUALITY CUT
df = df[df['Plx_corr']/df['Plx_err_corr'] > 3.0]
df = df.reset_index(drop = True)

# CORRECT PROPER MOTIONS
pmra_corrected, pmdec_corrected = [], []
for i in tqdm(range(len(df)), leave = False, desc = 'Correct proper motions'):
    pmra_corr, pmdec_corr = correct_proper_motion(pmra[i], pmdec[i], ra[i], dec[i],
                                                  phot_g_mean_mag[i])
    pmra_corrected.append(pmra_corr)
    pmdec_corrected.append(pmdec_corr)
df['pmra_corrected'] = pmra_corrected
df['pmdec_corrected'] = pmdec_corrected

# CORRECT PROPER MOTION ERRORS
df = correct_proper_motion_error(df, True)
pmra_error_corrected = df['pmra_error_corrected'].values
pmdec_error_corrected = df['pmdec_error_corrected'].values

# TRANSFORM INTO GALACTIC COORDINATES
l, b, pml, pmb, pml_error, pmb_error = [], [], [], [], [], []
for i in tqdm(range(len(df)), desc = 'Transform to Galactic', leave = False):
    vals, err, corr = galactic_proper_motion_full(ra[i], dec[i], ra_error[i], dec_error[i],
                                                  parallax[i], parallax_error[i], pmra_corrected[i],
                                                  pmdec_corrected[i], pmra_error_corrected[i],
                                                  pmdec_error_corrected[i],
                                                  ra_dec_corr[i], ra_parallax_corr[i],
                                                  ra_pmra_corr[i], ra_pmdec_corr[i],
                                                  dec_parallax_corr[i], dec_pmra_corr[i],
                                                  dec_pmdec_corr[i], parallax_pmra_corr[i],
                                                  parallax_pmdec_corr[i], pmra_pmdec_corr[i])
    l.append(vals[0])
    b.append(vals[1])
    pml.append(vals[3])
    pmb.append(vals[4])
    pml_error.append(np.hypot(err[3], 0.023)) # Apply the 0.023 systematic error in cuadrature after the coordinate transformation:
    pmb_error.append(np.hypot(err[4], 0.023))

df['l'] = l
df['b'] = b
df['pml'] = pml
df['pmb'] = pmb
df['pml_error'] = pml_error
df['pmb_error'] = pmb_error

# THEORETICAL PROPER MOTIONS AND RADIAL VELOCITY
l, b, r = df['l'].values, df['b'].values, df['Dist_P50'].values
Vr_th, pml_th, pmb_th = [], [], []
for i in tqdm(range(len(df)), leave = False, desc = 'Expected proper motions'):
    Vr_star, pml_star, pmb_star = expected_proper_motion(l[i], b[i], r[i])
    Vr_th.append(Vr_star)
    pml_th.append(pml_star)
    pmb_th.append(pmb_star)
df['Vr_th'] = Vr_th
df['pml_th'] = pml_th
df['pmb_th'] = pmb_th

# Convert to arrays
l, b, pml, pmb = np.array(l), np.array(b), np.array(pml), np.array(pmb)
pml_th, pmb_th = np.array(pml_th), np.array(pmb_th)
pml_error, pmb_error = np.array(pml_error), np.array(pmb_error)

# PECULIAR PROPER MOTION
df['pml_pec'] = pml-pml_th
df['pmb_pec'] = pmb-pmb_th
pm_pec = np.hypot(pml-pml_th, pmb-pmb_th)
pm_pec_err = np.hypot((pml-pml_th)*pml_error, (pmb-pmb_th)*pmb_error)/pm_pec
df['pm_pec'] = pm_pec
df['pm_pec_err'] = pm_pec_err

# SPEEDS
r_high = df['Dist_P84'].values
r_low = df['Dist_P16'].values
yr2sec = 3600*24*365
pc2km = 3.26156*299792.458*yr2sec
deg2rad = np.pi/180
radsec2masyr = (3600*1000/deg2rad)*yr2sec
conv_coeff = (pc2km/radsec2masyr)
df['Vl_pec'] = r*(pml-pml_th)*conv_coeff
df['Vb_pec'] = r*(pmb-pmb_th)*conv_coeff
Vpec = r*pm_pec*conv_coeff
sigma_r = (r_high-r_low)/2
sigma_Vpec = Vpec*np.hypot(sigma_r/r, pm_pec_err/pm_pec)
df['V_pec'] = Vpec
df['V_pec_err'] = sigma_Vpec
df['V_pec_low'] = Vpec-sigma_Vpec

# SAVE DATA
df.to_pickle('ALS_Sample.pkl')

# CREATE A PRELIMINARY RUNAWAY LIST AD SAVE IT
cols = ['ID_ALS', 'Name', 'Cat_ALS', 'Cat', 'SpT_ALS', 'STn', 'LCn',
        'b', 'Dist_P50', 'Vb_pec', 'Vl_pec', 'V_pec', 'V_pec_err', 'V_pec_low']
secondary_cols = [col for col in list(df.columns) if col not in cols]
dd = df.sort_values(by = 'V_pec', ascending = False)
dd = dd.reset_index(drop = True)
dd = dd[cols+secondary_cols]
dd = dd.replace('', np.nan)
dataframe_to_fits(dd, 'Preliminary_Runaways')

# DATA FOR JESÚS
djesus = dd[['ID_DR3', 'ID_ALS_Num', 'ID_ALS_Let', 'Name', 'ra', 'dec', 'l', 'b',
             'parallax', 'parallax_error', 'Plx_corr', 'Plx_err_corr', 'Dist_P16',
             'Dist_P50', 'Dist_P84', 'pmra', 'pmdec', 'pmra_error', 'pmdec_error',
             'pmra_corrected', 'pmdec_corrected', 'pmra_error_corrected',
             'pmdec_error_corrected', 'pml', 'pmb', 'pml_error', 'pmb_error',
             'pml_th', 'pmb_th', 'Vr_th', 'pml_pec', 'pmb_pec', 'Vl_pec',
             'Vb_pec', 'V_pec', 'V_pec_err', 'V_pec_low']]
djesus.to_csv('Data_for_Jesus.csv', sep = '\t', encoding = 'utf-8', index = False)
