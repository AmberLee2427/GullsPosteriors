import numpy as np
import sys
import os
import pandas as pd
from astropy import units as u
import astropy.coordinates as astrocoords
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from astropy.time import Time
from scipy.interpolate import interp1d
import pickle
import time
import warnings


class Data:

    def __init__(self):
        pass

    def new_event(self, path, sort='alphanumeric'):
        '''get the data and true params for the next event'''

        files = os.listdir(path)
        files = sorted(files)

        if path[-1] != '/':
            path = path + '/'

        if not os.path.exists(path+'emcee_run_list.txt'):  # if the run list doesn't exist, create it
            run_list = np.array([])
            np.savetxt(path+'emcee_run_list.txt', run_list, fmt='%s')

        if not os.path.exists(path+'emcee_complete.txt'):  # if the complete list doesn't exist, create it
            complete_list = np.array([])
            np.savetxt(path+'emcee_complete.txt', complete_list, fmt='%s')

        for file in files:
            if 'csv' in file:
                master_file = path + file

        if sort == 'alphanumeric':

            for file in files:

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    run_list = np.loadtxt(path+'emcee_run_list.txt', dtype=str)

                if (file not in run_list) and ('det.lc' in file):

                    print('Already ran:', run_list)
                    run_list = np.hstack([run_list, file])
                    print('Running:', file, type(file))
                    np.savetxt(path+'emcee_run_list.txt', run_list, fmt='%s')

                    lc_file_name = file.split('.')[0]
                    event_identifiers = lc_file_name.split('_')
                    EventID = event_identifiers[-1]
                    SubRun = event_identifiers[-3]  # the order of these is fucked up
                    Field = event_identifiers[-2]  # and this one. -A 2024-11-11 resample

                    data_file = path + file
                    
                    data = self.load_data(data_file)  # bjd, flux, flux_err, tshift, ushift
                    
                    event_name = f'{Field}_{SubRun}_{EventID}'
                    #print('event_name = ', event_name)

                    obs0_data = data[0].copy()
                    simt = obs0_data[7]
                    bjd = obs0_data[0]
                    
                    truths = self.get_params(master_file, EventID, SubRun, Field, simt, bjd)  
                    # turns all the degress to radians and sim time to bjd
                    break

        '''if ".txt" in sort:
            files = np.loadtxt(sort)
            for i in range(len(files)):
                if os.path.exists('runlist.npy'):
                    runlist = np.loadtxt('runlist.npy')
                else:
                    runlist = np.array([])
                if files[i] not in runlist:
                    runlist = np.vstack(files[i])
                    np.savetxt('runlist.txt', runlist, fmt='%s')
                    data = mm.MulensData(file_name='data/' + files[i])
                    true_params = np.loadtxt('true_params/' + files[i].split('.')[0] + '.txt')
                    break'''

        print()
        # this is fucking dumb, but the 'lcname's is the master file do not match the actual lc file names
        if file == truths['lcname']:  # check that the data file and true params match
            print('Data file and true params \'lcname\' match')
            sys.exit()
            #return event_name, truths, data
        else:
            print('Data file and true params \'lcname\' do not match')
            print(file, '!=', truths['lcname'])
            if len(file) != len(truths['lcname']):
                print('length:', len(file), '!=\n', len(truths['lcname']))
            return event_name, truths, data

    def load_data(self, data_file):
        r'''load the data file.
        
        Notes:
        ------
        The lightcurve columns are:
            [0] Simulation_time 
            [1] measured_relative_flux 
            [2] measured_relative_flux_error 
            [3] true_relative_flux
            [4] true_relative_flux_error 
            [5] observatory_code 
            [6] saturation_flag 
            [7] best_single_lens_fit 
            [8] parallax_shift_t
            [9] parallax_shift_u 
            [10] BJD 
            [11] source_x 
            [12] source_y 
            [13] lens1_x 
            [14] lens1_y 
            [15] lens2_x 
            [16] lens2_y

        Magnitudes can be computed using:
    
        ..math:
            m = m_{source} + 2.5 log f_s - 2.5 log{F}

        where :math:`F=fs*\mu + (1-fs)` is the relative flux (in the file), :math:`\mu` is the magnification, and
        
        ..math:
            \sigma_m = 2.5/ln{10} \sigma_F/F.

        These are listed in the header information in lines #fs and #Obssrcmag with order matching the observatory code order.
        The observatory codes correspond to 0=W146, 1=Z087, 2=K213

        Bugs/issues/caveats:
        The output file columns list a limb darkening parameter of Gamma=0, it is actually Gamma=0.36 (in all filters)
        The orbit for the Z087 observatory appears to be different to the W146 and K213 observatory
        Dev is working on producing the ephemerides, but for single observatory parallax, using interpolated versions of the ones
        available for the data challenge will probably be accurate enough, or an Earth ephemeris with the semimajor axis (but
        not period) increased by 0.01 AU
        Lenses with masses smaller than the isochrone grid limits (I believe 0.1 MSun, will have filler values for magnitudes
        and lens stellar properties).
        There may be some spurious detections in the list where the single lens fit failed. Please let dev know if you find any
        of these events so that we can improve the single lens fitter.'''

        header = ['Simulation_time', 
                  'measured_relative_flux', 
                  'measured_relative_flux_error', 
                  'true_relative_flux', 
                  'true_relative_flux_error', 
                  'observatory_code', 
                  'saturation_flag', 
                  'best_single_lens_fit', 
                  'parallax_shift_t', 
                  'parallax_shift_u', 
                  'BJD', 
                  'source_x', 
                  'source_y', 
                  'lens1_x', 
                  'lens1_y', 
                  'lens2_x', 
                  'lens2_y',
                  'A',
                  'B',
                  'C'
                  ]

        data = pd.read_csv(data_file, sep=r'\s+', skiprows=12, names=header)  # delim_whitespace=True is the same as sep=r'\s+', but older.
                                                                              # The 'r' in sep=r'\s+' means raw string, which is not necessary.
                                                                              # Otherwise you get annoying warnings.
        
        # simulation time to BJD
        print(data['BJD'][0])
        print(data['Simulation_time'][0])

        self.sim_time0 = np.sum(data['BJD'] - data['Simulation_time'])/len(data['BJD'])

        data = data[['BJD', 
                    'measured_relative_flux',
                    'measured_relative_flux_error',
                    'parallax_shift_t',
                    'parallax_shift_u', 
                    'observatory_code',
                    'true_relative_flux', 
                    'true_relative_flux_error',
                    'Simulation_time'
                    ]]
        
        data_dict = {}
        for code in data['observatory_code'].unique():
            data_obs = data[data['observatory_code'] == code][['BJD', 
                                          'measured_relative_flux', 
                                          'measured_relative_flux_error', 
                                          'parallax_shift_t', 
                                          'parallax_shift_u',
                                          'true_relative_flux', 
                                          'true_relative_flux_error',
                                          'Simulation_time'
                                          ]].reset_index(drop=True)
            data_dict[code] = data_obs.to_numpy().T

        return data_dict

    def get_params(self, master_file, EventID, SubRun, Field, epoch=None, bjd=None):
        '''get the true params for the event'''
        EventID = int(EventID)
        SubRun = int(SubRun)
        Field = int(Field)

        master = pd.read_csv(master_file, header=0, delimiter=',')
        #print(master.head())

        truths = master[(master['EventID'] == int(EventID)) & 
                        (master['SubRun'] == int(SubRun)) & 
                        (master['Field'] == int(Field))
                        ].iloc[0]

        #print(self.sim_time0)
        
        s = truths['Planet_s']
        q = truths['Planet_q']
        rho = truths['rho']
        u0 = truths['u0lens1']  # croin
        alpha = truths['alpha']*np.pi/180 # convert to radians
        if epoch is not None and bjd is not None:
            t0_sim = truths['t0lens1']
            t0_bjd = np.interp(t0_sim, epoch, bjd)
            t0 = t0_bjd
            tc_sim = truths['tcroin']
            tc_bjd = np.interp(tc_sim, epoch, bjd)
            tcroin = tc_bjd
        else:
            t0 = truths['t0lens1'] + self.sim_time0  # convert to BJD
            tcroin = truths['tcroin'] + self.sim_time0  # convert to BJDs
        truths['t0lens1'] = t0
        tE = truths['tE_ref'] 
        piEE = truths['piEE']
        piEN = truths['piEN']
        i = truths['Planet_inclination']*np.pi/180  # convert to radians
        phase = truths['Planet_orbphase']*np.pi/180  # convert to radians # centre on tcroin
        period = truths['Planet_period']*365.25  # convert to days
        #phase_change = truths['tcroin'] / period
        #phase = phase + phase_change  # centre on t0
        #phase = phase % (2.0*np.pi)  # make sure it's between 0 and 2pi
        truths['params'] = [s, q, rho, u0, alpha, t0, tE, piEE, piEN, i, phase, period]

        truths['tcroin'] = tcroin

        return truths