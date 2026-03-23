# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:34:28 2026

@author: jacob
Creates a synthetic light curve of RR Lyr for 4.11.2025. The template is based on several brightness measurements between 5.6.2025 
and 18.7.2025. I downloaded the data from AAVSO: https://www.aavso.org/data-download
Specifications:
    Object: RR Lyr
    Start date: 2460827.5
    End date: 2460888.5
    Discrepant data incuded: yes
    Diff and step data included: yes
    Format: comma delimited

https://www.aavso.org/LCGv2/ can be used to view the data
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from astroML.datasets import fetch_rrlyrae_templates
import matplotlib.pyplot as plt
from hjd2utc import hjd2utc
from datetime import datetime
from astropy.time import Time
from copy import deepcopy

from helper import get_repo_root

repo_root = get_repo_root()
#from plotting import plot
plt.close('all')

def utc2jd(utc):
    t_utc = Time(utc)
    t_jd = t_utc.jd
    return t_jd

def jd2utc(jd):
    t_jd = Time(jd, format='jd', scale='utc')
    return t_jd

def find_minmax_ind(mag, minmax='max'):
    #set NaNs to a high (dark) magnitude value
    mag_modified = deepcopy(mag)
    mag_modified[np.isnan(mag_modified)] = 100

    #find index of maximum
    if minmax == 'max':
        ind = int(np.where(mag_modified == np.min(mag_modified))[0][0])  #searches for the minimum, because at the brightness maximum, the magnitude is minimal
    elif minmax == 'min':
        ind = int(np.where(mag_modified == np.max(mag_modified))[0][0])
    return ind



rrlyrae = fetch_rrlyrae_templates()
g = rrlyrae["103g"][:, 1]
r = rrlyrae["103r"][:, 1]
t = rrlyrae["103g"][:, 0]
V = np.mean(np.array([g, r]), axis=0)

#plot(data=[t, V, g, r], legend=["V", "g", "r"])

fig = plt.figure()
plt.plot(t, V, 'k', label='V')
plt.plot(t, g, 'g', label='g')
plt.plot(t, r, 'r', label='r')
plt.gca().invert_yaxis()
plt.legend()
plt.show



#%% scale the light curve to match the true average period and the true magnitude
# (freq: f_0 from https://arxiv.org/pdf/1011.5908, page 7)

freq = 1.76416    #oscillations per day
period = 1 / freq #in days
t = t / np.max(t) * period

max_mag = 7.268149206349208     #obtained from AAVSO_lightcurve.py
min_mag = 8.17457142857143      #obtained from AAVSO_lightcurve.py
amplitude = abs(max_mag - min_mag)
offset = np.mean([max_mag, min_mag]) - .5  #-.5 because the original template is centered around .5
V = V*amplitude + offset


#%% chain several oscillations together starting with the nearest known brightness maximum
# (from https://rr-lyr.irap.omp.eu/dbrr/rrdb-V2.0_08.3.php?RR+Lyr& → this website lists the dates of known brightness maxima)

# find the brightness maximum that is closest to 4.11.2025
_ = hjd2utc(hjd=2460979.4531, ra_deg=291.36629, dec_deg=42.78436, print_result=True)  #→ 30.10.2025, 22:52:30.075
known_max_1 = datetime.strptime('30.10.2025, 22:52:30.075', '%d.%m.%Y, %H:%M:%S.%f')
_ = hjd2utc(hjd=2460995.3201, ra_deg=291.36629, dec_deg=42.78436, print_result=True)  #→ 15.11.2025, 19:41:59.154
known_max_2 = datetime.strptime('15.11.2025, 19:41:59.154', '%d.%m.%Y, %H:%M:%S.%f')

timediff = known_max_2 - known_max_1
n_periods = timediff.total_seconds() / 86400 / period   # = 27.993 → almost an exact multiple of the period → great!


# # third overview plot: two oscillations together
# fig_two_oscis = plt.figure()
# plt.plot(np.hstack((t, t+period)), np.hstack((V, V)), 'k')
# plt.gca().invert_yaxis()
# plt.show()


# chain oscillations between known_max_1 and known_max_2
n_periods = round(n_periods)
t_stack = []
V_stack = []
for n in range(n_periods):
    t_stack = np.append(t_stack, t + period*n)
    V_stack = np.append(V_stack, V)
t_stack += utc2jd(known_max_1)   #convert back to julian date



# # fourth overview plot: chained oscillations
# fig_stack = plt.figure(figsize=(20,8))
# plt.plot(t_axis_stack, mean_mag_stack, 'k')
# plt.gca().invert_yaxis()
# plt.show()



#%% cut out the light curve for 4.11.2025 and create a beautiful plot
start_jd = 2460983.5   #UTC 2025-11-04T00:00:00 = JD 2460983.5
start_utc = datetime(2025, 11, 4, 0, 0, 0)
end_jd = 2460984.5     #UTC 2025-11-05T00:00:00 = JD 2460984.5
end_utc = datetime(2025, 11, 5, 0, 0, 0)

inds_041125 = np.where((start_jd <= t_stack) & (t_stack < end_jd))[0]
t_axis_JD_041125 = jd2utc(t_stack[inds_041125]).iso
#t_axis_UTC_041125 = jd2utc(t_axis_JD_041125).to_datetime()
mean_mag_041125 = V_stack[inds_041125]


def model_lc(t, mag, save_csv=False):
    dti = pd.to_datetime(t)
    mag_fit = np.interp(dti, dti[np.isfinite(mag)], mag[np.isfinite(mag)])
    mag_df = pd.DataFrame(data={'t': dti, 'mag': mag, 'mag_fit': mag_fit})
    if save_csv:
        mag_df.to_csv(repo_root/'data/rr_lyr_model.csv')
    return mag_df

mag_df = model_lc(t_axis_JD_041125, mean_mag_041125, save_csv=True)

if __name__ == "__main__":
    # fifth plot: light curve of 4.11.2025
    fig_lightcurve = plt.figure(figsize=(12,8))
    #plt.plot(t_axis_JD_041125, mean_mag_041125, 'r')
    plt.plot(mag_df['t'], mag_df['mag'], 'r', label='mean')
    plt.plot(mag_df['t'], mag_df['mag_fit'], 'k-.', label='fit')
    plt.gca().invert_yaxis()
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.legend()

    plt.title('Synthetic Light Curve of RR Lyr of 4.11.2025 Filter: V')
    plt.xlabel('time')
    plt.ylabel('brightness / mag')
    plt.savefig(repo_root/'figs/rr_lyr_ref_V_band_full.png')

    plt.xlim([np.datetime64("2025-11-04 17:00"), np.datetime64("2025-11-04 22:00")])
    plt.ylim([8.12, 7.7])
    plt.savefig(repo_root/'figs/rr_lyr_ref_V_band_cut.png')

    # ticks_jd = np.arange(start_jd, end_jd, 1/12)  #one tick every 2 hours
    # tick_labels = jd2utc(ticks_jd).isot  #iso format 2025-11-04T02:00:00
    # plt.xticks(ticks_jd, tick_labels, rotation=45)

    #plt.show()












