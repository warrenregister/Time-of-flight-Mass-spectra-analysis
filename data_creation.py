"""
Helper methods for creating data necessary for atomic pattern recognizing
"""
from os import listdir
import numpy as np
import pandas as pd

def read_csvs(dir):
    """
    Read in all spectra csvs in the given directory to a dataframe containing
    values necessary for calibrating the spectrum.
    """
    info = ['Technique', 'StartFlightTime', 'Mass/Time', 'MassOffset',
            'SpecBinSize']
    data = [[] for name in info]
    channels = []
    intensities = []
    names = []
    for csv in listdir(dir):
        chans = []
        intens = []
        if csv[-3:] == 'csv':
            file = pd.read_csv(dir + csv, names=list('abcdefghijk'))
            names.append(csv)
            i = 0
            line = ''
            while 'EOFH' not in line:
                line = file.loc[i]['a']
                for j, name in enumerate(info):
                    if name in line:
                        datum = line.split(':')[-1]
                        try:
                            datum = float(datum)
                        except ValueError:
                            try:
                                datum = 0 if '-' in datum else 1
                                datum = int(datum)
                            except Exception as e:
                                print(e)
                        data[j].append(datum)
                i += 1
            for line in file.loc[i:].itertuples():
                try:
                    chans.append(float(line[1]))
                    intens.append(float(line[2]))
                except ValueError as e:
                    print(e)
                    print(line[1])
                    print(line[2])
            channels.append(chans)
            intensities.append(intens)
    return pd.DataFrame(
        list(zip(names, data[1], data[2], data[3], data[4], data[0],
                 channels, intensities)),
        columns=['file_name', 'StartFlightTime',
                 'MassOverTime', 'MassOffset', 'SpecBinSize',
                 'Technique', 'channels', 'intensities'])


def mass_formula(channels: np.array, spec_bin_size, start_time,  mass_over_time, mass_offset):
    '''
    Fast conversion from flightime to mass.
    '''
    return ((channels * .001 * spec_bin_size + start_time) * mass_over_time + mass_offset)**2


def generate_calibrated_data(data):
    '''
    Applies mass_formula to every row in dataset to allow
    calibrated graphs to be generated.
    '''
    new_data = data.copy()
    masses = []
    for row in new_data.itertuples():
        spec = row.SpecBinSize
        m_over_t = row.MassOverTime
        m_offset = row.MassOffset
        time = row.StartFlightTime
        masses.append(mass_formula(np.array(row.channels), spec, time,
                      m_over_t, m_offset))
    new_data['masses'] = masses
    return new_data


def get_isotope_data(path='../data/Elements.txt') -> pd.DataFrame:
    """
    Generate dataframe with isotope mass data for every element based on txt
    file with the information.
    """
    elements = []
    spots = []
    freqs = []
    with open(path) as file:
        for line in file.readlines():
            element_spots = []
            element_freqs = []
            sections = line.split('(')
            elements.append(sections[0].split()[2])
            for tup in sections[1:]:
                nums = tup.split(',')
                element_spots.append(float(nums[0]))
                element_freqs.append(float(nums[1].split(')')[0]))
            spots.append(element_spots)
            freqs.append(element_freqs)
    isotope_data = pd.DataFrame(list(zip(elements, spots, freqs)),
                                columns=['Element', 'IsotopeMasses',
                                         'IsotopeFrequencies'])
    return isotope_data