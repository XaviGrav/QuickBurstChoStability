#chack for updated files\n,
#import packages
from __future__ import division

import numpy as np
import glob, json
import pickle
import os as os_pack
import matplotlib.pyplot as plt
import corner
import healpy as hp
import os, glob, json, pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise_extensions import blocks
from enterprise_extensions import models as ee_models
from enterprise_extensions import model_utils as ee_model_utils
from enterprise_extensions import model_orfs
from enterprise_extensions.hypermodel import HyperModel
from enterprise_extensions.frequentist import optimal_statistic as opt_stat
from enterprise_extensions import sampler as ee_sampler
from enterprise.signals.signal_base import LogLikelihood
#import enterprise_wavelets as models
from enterprise.signals.deterministic_signals import Deterministic
from enterprise.signals.parameter import function

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import re
import sys
sys.path.append("/home/mitch/QuickBurstWork/QuickBurstChoStability/")
from QuickBurst import QuickBurst_MCMC as QuickBurst_MCMC


with open("/home/mitch/pulsar_data/15_year_data/psrs_trimmed_SNR99p.pkl", 'rb') as f:
    psrs_total = pickle.load(f)

noise_file = "/home/mitch/pulsar_data/15year_data_pkl_quickburst/v1p1_all_dict.json"

with open(noise_file, 'r') as h:
    noise_params = json.load(h)

#Setting dataset max time and reference time
maximum = 0
minimum = np.inf
for psr in psrs_total:
    if psr.toas.max() > maximum:
        maximum = psr.toas.max()
    if psr.toas.min() < minimum:
        minimum = psr.toas.min()

#Sets reference time
tref = minimum
print(tref/3600/24/365)
t0_max = (maximum - minimum)/365/24/3600
print(t0_max)

#GET A SINGLE PULSAR TO RUN
psr_name = "J1614-2230"
for psr in psrs_total:
    if psr.name == psr_name:
        psrs = [psr]
        break

#GET A RANGE OF PULSARS    
#psrs= psrs_total[0:42]

print(psrs[0].name)

#Number of shape parameter updates
N_slow=int(1e5)

#How often to update fisher matrix proposals (based on shape parameter updates)
n_fish_update = int(N_slow/10)

#Ratio of projection parameter updates per shape parameter update
projection_updates = 10000

#Proposal weights (must sum to 1)
DE_prob = 0.3
fisher_prob = 0.6 
prior_draw_prob = 0.1

#Number of samples to thin (based on total samples N_slow*projection_updates)
thinning = projection_updates

T_max = 4 #2
n_chain = 5 #3

#Prior bounds on shape params
tau_min = 0.05
tau_max = 5.0 #3.0
f_max = 1e-7
f_min = 3.5e-9 #1e-8

#Load in tau scan proposal files
ts_file = "/home/mitch/QuickBurstWork/QuickBurstChoStability/tau_scans/wavelets/PTA_tauscan.pkl"

# resuming_pulsar = 'J0030+0451'
for psr in psrs:
    glitch_ts_file = f"/home/mitch/QuickBurstWork/QuickBurstChoStability/tau_scans/transients/{psr.name}/transients.pkl"
    filepath = "/home/mitch/QuickBurstWork/QuickBurstChoStability/chains/{}/".format(psr.name)
    print('Starting pulsar {}'.format(psr.name))
    os.makedirs(filepath, exist_ok = True)
    savefile = "chain_1"
    savepath = filepath + savefile #NOTE: DO NOT ADD FILE EXTENSION
    
    model_labels = []
    #Tracking run settings
    model_labels.append(["max 0 wavelets ", "max 3 glitches ", "projection_updates/shape_update=10000", "thinning=10000",  "jumps: glitch rj, glitch tau, PT swap, fast, noise", 
                        "noise jump weight (DE, fish, prior) = (0.3,0.6,0.1)"])
    #Prior information
    model_labels.append(["Uniform wavelet amp priors: [-10, -5]"])
    
    with open(filepath+'/run_info.json' , 'w') as fout:
        json.dump(model_labels, fout, sort_keys=True,
                indent=4, separators=(',', ': '))

    #samples, acc_fraction, swap_record, rj_record, ptas, log_likelihood, betas, PT_acc
    _, _, _, _, _, _, _, _ = QuickBurst_MCMC.run_qb(N_slow, T_max, n_chain, [psr,],
                                                                        max_n_wavelet=3,
                                                                        min_n_wavelet=0,
                                                                        n_wavelet_start=0,
                                                                        RJ_weight=1,
                                                                        glitch_RJ_weight=1,
                                                                        regular_weight=1,
                                                                        noise_jump_weight=2,
                                                                        PT_swap_weight=1,
                                                                        tau_scan_proposal_weight=1,
                                                                        glitch_tau_scan_proposal_weight=1,
                                                                        DE_prob = DE_prob,
                                                                        fisher_prob = fisher_prob,
                                                                        prior_draw_prob = prior_draw_prob,
                                                                        de_history_size = 2500,
                                                                        tau_scan_file=ts_file,
                                                                        glitch_tau_scan_file=glitch_ts_file,
                                                                        #gwb_log_amp_range=[-18,-15],
                                                                        rn_log_amp_range=[-18,-11],
                                                                        wavelet_log_amp_range=[-10.0,-5.0],
                                                                        per_psr_rn_log_amp_range=[-18,-11],
                                                                        #rn_params = [noise_params['gw_crn_log10_A'],noise_params['gw_crn_gamma']],
                                                                        prior_recovery=False,
                                                                        #gwb_amp_prior='log-uniform',
                                                                        rn_amp_prior='log-uniform',
                                                                        wavelet_amp_prior='uniform',
                                                                        per_psr_rn_amp_prior='log-uniform',
                                                                        #gwb_on_prior=0.975,
                                                                        max_n_glitch=3,
                                                                        #n_glitch_start='random',
                                                                        glitch_log_amp_range=[-10.0,-5.0],
                                                                        glitch_amp_prior='uniform',
                                                                        f0_max = f_max,
                                                                        f0_min = f_min,
                                                                        tau_max_in = tau_max,
                                                                        tau_min_in = tau_min,
                                                                        t0_max = (psr.toas.max()-minimum)/24/365/3600, #years
                                                                        t0_min = (psr.toas.min()-minimum)/24/365/3600, #years
                                                                        tref = (minimum), #seconds
                                                                        vary_white_noise=True,  
                                                                        include_rn=False, vary_rn=False,
                                                                        include_equad=True,
                                                                        include_ecorr=True,
                                                                        include_efac=True,
                                                                        wn_backend_selection=True,
                                                                        noisedict = noise_params,
                                                                        include_per_psr_rn=True,
                                                                        vary_per_psr_rn=True,
                                                                        # resume_from=savepath,
                                                                        #per_psr_rn_start_file=RN_start_file,
                                                                        n_fish_update = n_fish_update,
                                                                        savepath=savepath, save_every_n=100,
                                                                        n_fast_to_slow=projection_updates, thin = thinning)