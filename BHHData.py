# Import required function
import os
import urllib2
import numpy as np
import matplotlib.pyplot as plt
from pycbc.frame.losc import losc_frame_urls
from pycbc.frame import read_frame
from pycbc.waveform import get_td_waveform
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.filter import sigma, matched_filter
import json


def download_from(url, filepath='./', filename=None):
    """ Downloads a file from the given url.
    filepath defines where to save the file
    if filename is not given then it is taken from the url."""
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if filename is None:
        filename = url.split('/')[-1]
    if os.path.exists(filepath + filename):
        print('File already downloaded')
    else:
        print('Starting Download from ' + url)
        data = urllib2.urlopen(url).read()
        with open(filepath + filename, 'w') as f:
            f.write(data)
        print('Download Complete')
    return filepath + filename


def split_by_true(strain, condition):
    """This function takes a PyCBC strain segment and splits it 
    into a list of continuous strain segments removing the 
    sections that meet the condition."""
    trues = True
    strains = []
    while trues is True:
        if (True in condition) and not condition[0]:
            i = np.argmax(condition)
            strains.append(strain[:i])
            strain = strain[i:]
            condition = condition[i:]
        elif (True in condition) and (False in condition):
            i = np.argmin(condition)
            strain = strain[i:]
            condition = condition[i:]
        elif not True in condition:
            strains.append(strain)
            trues = False
        else:
            trues = False
    return strains


def create_all_noise(bank, ifos, minnoise=10, band=None, seconds=4, filepath='./'):
    """ This function creates noise files from the dictionary of urls provided.
    bank is the dictionary containing the noise and signal data.
    ifos is the list of interferometers
    urls is a dictionary with the ifos as their keys listing urls for frame files
    minnoise is the minimum number of files needed
    band is a list with two elements [low f cut, high f cut]
    seconds is the length of the noise files required
    downloadpath is the path where frame files should be downloaded to"""
    if not os.path.exists('./noiseBank'):
        os.makedirs('./noiseBank')
    if bank['nnoises'] >= minnoise:
        print('{0} noise files already exist. Noise file creation complete'.format(nnoises))
        return bank
    elif bank['nnoises'] < minnoise:
        print('{0} noise files already exist. {1} will be created'.format(bank['nnoises'], minnoise-bank['nnoises']))
    times = {ifo:[] for ifo in ifos}
    for ifo in ifos:
        times[ifo] = [[int(url.split('-')[-2]), int(url.split('-')[-2])+4096] for url in bank['urls'][ifo]]
    ifoi, urli = 0, 0
    start = False
    while start is False:
        url = str(bank['urls'][ifos[ifoi]][urli])
        time = times[ifos[ifoi]][urli]
        if time[1] > bank['noise_checked'][ifos[ifoi]][1]:
            print('Data is available from {0}. Reading data...'.format(url))
            start = True
        elif urli == len(bank['urls'][ifos[ifoi]])-1 and ifoi == len(ifos)-1:
            print('No more data in available for cutting. {0} noise files exist.'.format(bank['nnoises']))
            return bank
        elif urli == len(bank['urls'][ifos[ifoi]])-1:
            print('No more data is available for {0}. Trying next IFO'.format(ifos[ifoi]))
            urli = 0
            ifoi += 1
        else:
            print('No more data is available from {0}. Trying next url'.format(url))
            urli += 1
    ifos = ifos[ifoi:]
    urls = {ifo:bank['urls'][ifo] for ifo in ifos}
    urls[ifos[0]] = urls[ifos[0]][urli:]
    for ifo in ifos:
        for url in urls[ifo]:
            if not url in bank['glitch_checked'][ifo]:
                bank, strains = find_glitches(bank, ifo, url, band=band, seconds=seconds, filepath=filepath)
            else:
                file = download_from(url, filepath=filepath)
                file_strain = read_frame(file, ifo+':LOSC-STRAIN')
                strains = split_by_true(file_strain, np.isnan(file_strain.numpy()))
            print('Cutting noise files from {0}'.format(url))
            for strain in strains:
                if strain.sample_times[-1] > bank['noise_checked'][ifo][1]:
                    sr = strain.sample_rate
                    strain = strain.whiten(4., 4., remove_corrupted=False)
                    if band is not None:
                        strain = strain.highpass_fir(band[0], 512, remove_corrupted=False).lowpass_fir(band[1], 512, remove_corrupted=False)
                    possible = int(np.floor(1.*(len(strain) - 16*sr)/sr/seconds))
                    chunkis = [[8*sr+i*seconds*sr, 8*sr+(i+1)*seconds*sr] for i in range(possible)]
                    for chunki in chunkis:
                        if strain.sample_times[chunki[1]] > bank['noise_checked'][ifo][1]:
                            chunk = strain[chunki[0]:chunki[1]]
                            bank['noise_checked'][ifo][1] = strain.sample_times[chunki[1]]
                            if not strain.sample_times[chunki[0]] in bank['glitches'][ifo]:
                                bank['nnoises'] += 1
                                chunk.save_to_wav('./noiseBank/noise_{0}.wav'.format(bank['nnoises']))
                                plt.figure(figsize=[16, 6])
                                plt.plot(np.arange(len(chunk))/4096., chunk)
                                plt.savefig('./noiseBank/noise_{0}.png'.format(bank['nnoises']))
                                plt.close()
                        if bank['nnoises'] >= minnoise:
                            print('Required file have been created. There are {0} noise files.'.format(bank['nnoises']))
                            return bank


def find_glitches(bank, ifo, url, band=None, seconds=4, filepath='./'):
    print('Checking {0} for glitches and cutting.'.format(url))
    if not os.path.exists('./noiseBank'):
        os.makedirs('./noiseBank')
    file = download_from(url, filepath=filepath)
    file_strain = read_frame(file, ifo+':LOSC-STRAIN')
    strains_raw = split_by_true(file_strain, np.isnan(file_strain.numpy()))
    strains = [strain.copy() for strain in strains_raw]
    masses = [20, 50, 100]
    hps = []
    for i in range(len(masses)):
        hp, _ = get_td_waveform(approximant='IMRPhenomD', mass1=masses[i], mass2=masses[i], delta_t=1./4096, f_lower=30)
        hps.append(hp)
    for strain in strains:
        sr = strain.sample_rate
        psd = strain.psd(4.)
        psd = interpolate(psd, strain.delta_f)
        psd = inverse_spectrum_truncation(psd, 4*sr, low_frequency_cutoff=20)
        snrs = []
        for i in range(len(masses)):
            template = hps[i].copy()
            template.resize(len(strain))
            template = template.cyclic_time_shift(template.start_time)
            snr = matched_filter(template, strain, psd=psd, low_frequency_cutoff=20)
            snrs.append(snr)
        strain = (strain.to_frequencyseries()/psd**0.5).to_timeseries()
        if band is not None:
            strain = strain.highpass_fir(band[0], 512, remove_corrupted=False).lowpass_fir(band[1], 512, remove_corrupted=False)
        possible = int(np.floor(1.*(len(strain) - 16*sr)/sr/seconds))
        chunkis = [[8*sr+i*seconds*sr, 8*sr+(i+1)*seconds*sr] for i in range(possible)]
        for chunki in chunkis:
            snr_chunk = [snr.numpy()[chunki[0]:chunki[1]] for snr in snrs]
            if np.max(snr_chunk) > 5:
                chunk = strain[chunki[0]:chunki[1]]
                bank['nglitches'] += 1
                bank['glitches'][ifo].append(strain.sample_times[chunki[0]])
                chunk.save_to_wav('./noiseBank/glitch_{0}.wav'.format(bank['nglitches']))
                plt.figure(figsize=[16, 6])
                plt.plot(np.arange(len(chunk))/4096., chunk)
                plt.savefig('./noiseBank/glitch_{0}.png'.format(bank['nglitches']))
                plt.close()
    bank['glitch_checked'][ifo].append(url)
    print('Glitch file creation complete. There are {0} glitch files.'.format(bank['nglitches']))
    return bank, strains_raw


def create_all_glitches(bank, ifos, band=None, filepath='./'):
    ifoi, urli = 0, 0
    while bank['nglitches'] < minglitches:
        ifo, url = ifos[ifoi], bank['urls'][ifos[ifoi]][urli]
        if not url in bank['glitch_checked'][ifo]:
            bank, strains = find_glitches(bank, ifo, url, band=band, filepath=filepath)
        elif urli == len(bank['urls'][ifos[ifoi]])-1 and ifoi == len(ifos)-1:
            print('No more data in available for cutting. {0} glitch files exist.'.format(bank['nnglitches']))
            break
        elif urli == len(bank['urls'][ifos[ifoi]])-1:
            print('No more data is available for {0}. Trying next IFO'.format(ifos[ifoi]))
            urli = 0
            ifoi += 1
        else:
            print('No more data is available from {0}. Trying next url'.format(url))
            urli += 1
    return bank


def create_waveform(bank, ifo, strain, snr_levels, approximant, variables, change_inc=False, change_position=False, band=None, seconds=4):
    """ This functions takes a strain segment for use as noise, the snrs that the signal should be,
    the approximant to be used for the waveform, the variables to be used for masses and spins,
    and optionally, change_inc if True will vary the inclination, change_position if True will vary the sky position,
    band should be a list with the first element being the low frequency cut off and the second being the high frequency cut off,
    seconds is the length of the segments to be output in seconds.
    This will then save a .wav and .png file of the signal waveform and the signal within noise at all snrs"""
    # Unpack variables and create the waveform
    sr = strain.sample_rate
    mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = variables
    hp, hc = get_td_waveform(approximant=approximant, mass1=mass1, mass2=mass2,
                             spin1x=spin1x, spin1y=spin1y, spin1z=spin1z,
                             spin2x=spin2x, spin2y=spin2y, spin2z=spin2z,
                             delta_t=1./4096, f_lower=30)
    # crop the template to a the desired length
    hp = hp.crop(0, hp.end_time - 0.1)
    hp = hp[max(0, len(hp)-(seconds+1)*sr):]
    hc = hc.crop(0, hc.end_time - 0.1)
    hc = hc[max(0, len(hc)-(seconds+1)*sr):]
    if change_inc is True:
        # if change_inc is True, scale hp and hc accordingly
        inc_deg = np.random.randint(0, 91)
        inc = np.round(inc_deg*np.pi/180, 2)
        hp = hp*(1+np.cos(inc)**2)/2
        hc = hc*np.cos(inc)
    if change_position is True:
        # if change_position is True, calculate the response of the detector and apply it
        theta_deg, phi_deg, psi_deg = np.random.randint(0, 91), np.random.randint(0, 181), np.random.randint(0, 181)
        theta, phi, psi = np.round(theta*np.pi/180, 2), np.round(phi*np.pi/180, 2), np.round(psi*np.pi/180, 2)
        fp = -0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.cos(2*psi) - np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)
        fc = 0.5*(1+np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi) - np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)
        h = fp*hp + fc*hc
    else:
        # otherwise assume theta and psi = 0 and phi = pi/2, such that fp = 1 and fc = 0
        h = hp.copy()
    origin = ('{0}_m1_{1}_m2_{2}_1x_{3}_1y_{4}_1z_{5}_2x_{6}_2y_{7}_2z_{8}'
              .format(approximant, variables[0], variables[1], variables[2], variables[3],
                      variables[4], variables[5], variables[6], variables[7]))
    if len(h) > seconds*sr:
        h[:len(h)-seconds*sr] *= np.hanning(2*(len(h)-seconds*sr))[:len(h)-seconds*sr]
    # select a random index for the signal start, avoid the first and last 128 seconds to allow for cut
    avoided = False
    glitches = bank['glitches'][ifo]
    while avoided is False:
        eventi = int(128*sr + np.random.rand()*(len(strain) - len(h) - 256*sr))
        zone = [strain.sample_times[eventi-4*sr], strain.sample_times[eventi+4*sr]]
        inzone = 0
        for glitch in glitches:
            if zone[0] <= glitch <= zone[1]:
                inzone += 1
        if inzone == 0:
            avoided = True
    # add signal to noise and cut data to 256 seconds centred on event
    data = strain.copy()
    data[eventi:eventi+len(h)] += h.numpy()
    data = data[eventi-128*sr:eventi+128*sr]
    noise = strain[eventi-128*sr:eventi+128*sr]
    # find the PSD and use it to find the sigma value for the template
    psd = noise.psd(4)
    psd = interpolate(psd, noise.delta_f)
    psd = inverse_spectrum_truncation(psd, sr*4, low_frequency_cutoff=30)
    htemp = h.copy()
    htemp.resize(len(data))
    sgm = sigma(htemp, psd=psd, low_frequency_cutoff=30)
    # find the SNR of the template with detector noise
    nht = matched_filter(htemp, noise, psd=psd, low_frequency_cutoff=30)
    nh = nht[128*sr]
    # scale the template to the distance required to get the required snr and add to noise
    distances = []
    for snr_level in snr_levels:
        distances.append(1.*sgm/np.abs(snr_level[0]-nh))
    h_chunk = h.copy()[max(0, len(h)-seconds*sr):]
    if not os.path.exists('./signalBank/sigbank{0}'.format(bank['nsignals']+1)):
        os.makedirs('./signalBank/sigbank{0}'.format(bank['nsignals']+1))
    plt.figure(figsize=[16, 6])
    plt.plot(np.arange(1.*len(h_chunk))/sr, h_chunk.numpy())
    plt.xlim([0, 1.*(len(h_chunk)-1)/sr])
    plt.ylim([np.min(h_chunk.numpy())*1.2, np.max(h_chunk.numpy())*1.2])
    plt.savefig('./signalBank/sigbank{0}/Waveform.png'.format(bank['nsignals']+1))
    plt.close()
    h_chunk.save_to_wav('./signalBank/sigbank{0}/Waveform.wav'.format(bank['nsignals']+1))
    for snr_level, distance in zip(snr_levels, distances):
        hout = h.copy()*1./distance
        dataout = strain.copy()
        dataout[eventi:eventi+len(hout)] += hout.numpy()
        dataout = dataout[eventi-128*sr:eventi+128*sr]
        # whiten, bandpass and pitch shift data for better listening
        dataout = (dataout.to_frequencyseries()/psd**0.5).to_timeseries()
        if band is not None:
            dataout = dataout.highpass_fir(band[0], 512, remove_corrupted=False).lowpass_fir(band[1], 512, remove_corrupted=False)
        # crop the data so the signal is at a random psoition in the cut, avoiding the merger in the first 1 and last 0.25 seconds
        shift = int(seconds/2.*sr + np.random.rand()*(seconds-0.25-seconds/2.)*sr)
        data_chunk = dataout[128*sr+len(hout)-shift:128*sr+len(hout)-shift+seconds*sr]
        # Plot and save figures
        plt.figure(figsize=[16, 6])
        plt.plot(np.arange(len(data_chunk))/4096., data_chunk.numpy())
        plt.xlim([0, 1.*(len(data_chunk.numpy())-1)/sr])
        plt.ylim([np.min(data_chunk.numpy())*1.2, np.max(data_chunk.numpy())*1.2])
        plt.savefig('./signalBank/sigbank{0}/level_{1}.png'.format(bank['nsignals']+1, snr_level[1]))
        plt.close()
        # Save .wav files
        data_chunk.save_to_wav('./signalBank/sigbank{0}/level_{1}.wav'.format(bank['nsignals']+1, snr_level[1]))
    bank['nsignals'] += 1
    bank['info{0}'.format(bank['nsignals'])] = {}
    bank['info{0}'.format(bank['nsignals'])]['origin'] = origin
    bank['info{0}'.format(bank['nsignals'])]['mass1'] = variables[0]
    bank['info{0}'.format(bank['nsignals'])]['mass2'] = variables[1]
    bank['info{0}'.format(bank['nsignals'])]['spin1'] = (variables[2], variables[3], variables[4])
    bank['info{0}'.format(bank['nsignals'])]['spin2'] = (variables[5], variables[6], variables[7])
    bank['info{0}'.format(bank['nsignals'])]['system'] = 'Compact Binary Coalescence'
    if change_inc is True:
        bank['info{0}'.format(bank['nsignals'])]['inclination'] = inc_deg
    else:
        bank['info{0}'.format(bank['nsignals'])]['inclination'] = '0'
    if change_position is True:
        bank['info{0}'.format(bank['nsignals'])]['position'] = (theta_deg, phi_deg, psi_deg)
    else:
        bank['info{0}'.format(bank['nsignals'])]['position'] = ('0', '90', '0')
    return bank


def create_all_waveforms(bank, ifos, snrs, requested, minsignals=10, change_inc=False, change_position=False, band=None, filepath='./'):
    origin_list = [bank['info'+str(i)]['origin'] for i in range(1, bank['nsignals']+1)]
    snr_levels = []
    for i in range(0, len(snrs)):
        snr_levels.append([snrs[i], i+1])
    ifo = ifos[np.random.randint(0, len(ifos))]
    while len(bank['glitch_checked'][ifo]) == 0:
        ifo = ifos[np.random.randint(0, len(ifos))]
    selector = np.random.randint(0, len(bank['glitch_checked'][ifo]))
    select = bank['glitch_checked'][ifo][selector]
    file = download_from(select, filepath=filepath)
    file_strain = read_frame(file, ifo+':LOSC-STRAIN')
    strains_pre = split_by_true(file_strain, np.isnan(file_strain.numpy()))
    strains = [strain for strain in strains_pre if len(strain) > 256*4096]
    while bank['nsignals'] < minsignals:
        each = int(np.ceil(1.*(minsignals-bank['nsignals'])/np.sum([len(requested[approximant]) for approximant in requested])))
        for approximant in requested.keys():
            for listnum, space in enumerate(requested[approximant]):
                # unpack variables and create lists for combinations
                masses, maxq, minm, maxm = space[0:4]
                xspins, yspins, zspins, minspin, maxspin = space[4:9]
                masspairs = []
                spins = []
                fullset = []
                # loop over masses making sure they fall within the models restrictions and m1 is the larger mass, save valid mass pairs
                for i in range(len(masses)):
                    for j in range(i, len(masses)):
                        m1, m2 = masses[j], masses[i]
                        if (1.*m1/m2 <= maxq) and (minm <= m1+m2 <=maxm):
                            masspairs.append([m1, m2])
                # loop over spins and check the combined spin is less than 1, save the valid sets
                for spin1x in xspins:
                    for spin1y in yspins:
                        for spin1z in zspins:
                            if (spin1x**2 + spin1y**2 + spin1z**2)**0.5 < 1:
                                spins.append([spin1x, spin1y, spin1z])
                # loop over mass and spin sets, making sure the effective spin is within the model range and save final list
                for ms in masspairs:
                    for i in range(len(spins)):
                        for j in range(len(spins)):
                            sp1, sp2 = spins[i], spins[j]
                            chieff = (ms[0]*sp1[2] + ms[1]*sp2[2])/(ms[0] + ms[1])
                            if (ms[0]==ms[1] and i >= j) or ((ms[0] != ms[1]) and (minspin <= chieff <= maxspin)):
                                fullset.append([ms[0], ms[1], sp1[0], sp1[1], sp1[2], sp2[0], sp2[1], sp2[2]])
                number = len(fullset)
                print(approximant, number)
                # if there are more than the required amount combinations then select them at random
                if number > each:
                    selected = np.random.choice(len(fullset), size=each, replace=False)
                    number = each
                else:
                    selected = np.arange(number)
                # check that the files do not already exist, if they don't, use the function create_waveforms to save the files
                count = 1
                for i in selected:
                    variables = fullset[i]
                    origin = ('{0}_m1_{1}_m2_{2}_1x_{3}_1y_{4}_1z_{5}_2x_{6}_2y_{7}_2z_{8}'
                            .format(approximant, variables[0], variables[1], variables[2], variables[3],
                                    variables[4], variables[5], variables[6], variables[7]))
                    if origin in origin_list:
                        print('{0}/{1} - {2} list {3}  variables = {4}  Already exists'.format(count, number, approximant, listnum+1, variables))
                    else:
                        try:
                            strain = strains[np.random.randint(0, len(strains))]
                            bank = create_waveform(bank, ifo, strain, snr_levels, approximant, variables, band=band, change_inc=change_inc, change_position=change_position)
                        except RuntimeError:
                            print('{0}/{1} - {2} list {3}  variables = {4}  Failed'.format(count, number, approximant, listnum+1, variables))
                        else:
                            print('{0}/{1} - {2} list {3}  variables = {4}  Succesfull'.format(count, number, approximant, listnum+1, variables))
                            origin_list.append(origin)
                    count += 1
    return bank


ifos = ['H1', 'L1']
O1times = [1126051217, 1137254417]
filepath = './O1Data/'

# Creat a dictionary with the approximant, the component masses, max mass ratio, min mass, max mass, component spins, min and max spin for unequal masses
"""
requested = {'EOBNRv2HM': [np.arange(5, 25, 5), 6, 24, 1000, [0], [0], [0], -0, 0], 
             'IMRPhenomD': [np.arange(1, 11, 1), 18, 0, 1000, [0], [0], [-0.8, -0.5, 0, 0.5, 0.8], -0.85, 0.85],
             'IMRPhenomPv2': [np.arange(1, 11, 1), 5, 0, 1000, [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0, 0.5], -0.85, 0.85],
             'SEOBNRv3_opt': [np.round(np.arange(1, 2.1, .1), decimals=1), 8, 0, 65, [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0, 0.5], -0.7, 0.85],
             'SEOBNRv4_opt': [np.arange(5, 25, 5), 10, 9, 1000, [0], [0], [-0.8, -0.5, 0, 0.5, 0.8], -0.7, 0.85]}
"""
requested = {'EOBNRv2HM': [[np.arange(5, 25, 5), 6, 24, 1000, [0], [0], [0], -0, 0]], 
             'IMRPhenomD': [[np.round(np.arange(1, 2.1, .1), decimals=1), 18, 0, 1000, [0], [0], [0], -0.85, 0.85],
                            [np.round(np.arange(1, 2.1, .1), decimals=1), 18, 0, 1000, [0], [0], [-0.8, -0.5, 0, 0.5, 0.8], -0.85, 0.85],
                            [np.arange(1, 11, 1), 18, 0, 1000, [0], [0], [0], -0.85, 0.85],
                            [np.arange(1, 11, 1), 18, 0, 1000, [0], [0], [-0.8, -0.5, 0, 0.5, 0.8], -0.85, 0.85]],
             'IMRPhenomPv2': [[np.round(np.arange(1, 2.1, .1), decimals=1), 5, 0, 1000, [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0, 0.5], -0.85, 0.85],
                              [np.arange(1, 11, 1), 5, 0, 1000, [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0, 0.5], -0.85, 0.85]]}

minsignals, minnoises, minglitches = 10, 10, 1
snrs = [50, 40, 35, 30, 25, 22, 20, 18, 16, 14]
band = [20, 2000]

if os.path.isfile('./signalBank.json'):
    with open('./signalBank.json', 'r') as f:
        bank = json.load(f)
else:
    bank = {}
    bank['nsignals'] = 0
    bank['nnoises'] = 0
    bank['nglitches'] = 0
    bank['noise_checked'] = {ifo:[O1times[0], O1times[0]] for ifo in ifos}
    bank['glitch_checked'] = {ifo:[] for ifo in ifos}
    bank['glitches'] = {ifo:[] for ifo in ifos}
    bank['urls'] = {ifo:losc_frame_urls(ifo, O1times[0], O1times[1]) for ifo in ifos}

bank = create_all_noise(bank, ifos, minnoise=minnoises, band=band, filepath=filepath)

bank = create_all_glitches(bank, ifos, band=band, filepath=filepath)

bank = create_all_waveforms(bank, ifos, snrs, requested, minsignals=minsignals, change_inc=True, change_position=False, band=band, filepath=filepath)

with open('./signalBank.json', 'w') as f:
    json.dump(bank, f)
