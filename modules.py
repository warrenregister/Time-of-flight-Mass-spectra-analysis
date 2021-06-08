# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:51:15 2020

@author: Amber
"""

import pandas as pd
import os
import re
import numpy as np
import pickle

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import matplotlib.ticker as tck
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter, find_peaks
import pandas as pd


def peak_finder(mass, dis, h, p):
    """
    Uses scipy.signal.find_peaks to create a new dataframe of peaks matching the
    passed parameters for find_peaks. Returns this new dataframe.
    Arguments -------
    mass: dataframe containing m / z and intensity for peaks in spectra
    dis: Inter-Peak Distance parameter for scipy peak finder
    h: Height parameter for scipy peak finder
    p: prominence parameter for scipy peak finder
    """

    # set matplotlib global parameters
    plt.style.use('seaborn-white')
    # plt.style.use('ggplot')  its nice!
    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 25
    plt.rcParams['axes.labelsize'] = 25
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['ytick.labelsize'] = 25
    plt.rcParams['legend.fontsize'] = 25
    plt.rcParams['figure.titlesize'] = 25

    # get tof spectra details
    x = mass.iloc[:, 0][1000:200000]  # mass probably
    counts = mass.iloc[:, 1][1000:200000]  # intensity probably
    y = np.log(mass.iloc[:, 1])[1000:200000]  # nat log intensity probably

    # scipy.signal.find_peaks call, uses log intensity as input
    # stores indices of 'found peaks' in peaks
    peaks, _ = find_peaks(y, distance=dis, height=h, prominence=p)

    # graph mass and nat log intensity highlighting found peaks with red dots
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(x, y, color='k')  # plot peaks
    ax.scatter(x.iloc[peaks], y.iloc[peaks], color='r', s=20)  # plot red dots
    plt.xlabel('Mass-to-charge ratio [m/z[)]')
    plt.ylabel('Count [log]')
    ax.minorticks_on()
    ax.tick_params(direction='out', axis='both', which='major', length=6,
                   width=2, colors='black',
                   grid_color='black', grid_alpha=0.5)
    ax.tick_params(direction='out', axis='both', which='minor', length=4,
                   width=2, colors='black')
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # For the minor ticks, use no labels; default NullFormatter.
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    plt.xlim(0, max(x.iloc[peaks] + 10))
    plt.ylim(0, 15)
    plt.grid(linestyle='dotted')

    y = y.reset_index(drop=True)
    new_mass = pd.concat([x.iloc[peaks], counts.iloc[peaks]], axis=1,
                         ignore_index=True, sort=False)
    new_mass.columns = ['m', 'count']
    return new_mass


class Atomic_pattern_recognizer(object):

    def __init__(self, path):
        """
        Initiate APR. Define names of elements/molecules associated with
        each model, and load models from their file paths.

        Arguments -------
        path: full path to folder with models in it
        """
        self.three_peaks_names = ['Mg', 'S', 'Si', 'Fe', 'Ca_3P', 'double_S',
                                  'double_B', 'Random_class_three', 'Ni_3P',
                                  'Cr_3P']
        self.four_peaks_names = ['Cr', 'Zn', 'Fe', 'Ni', 'Zr', 'Ca',
                                 'Random_class_four']
        self.five_peaks_names = ['Ti', 'Ni', 'Ge', 'Zn', 'Se', 'Zr',
                                 'Random_class_five']
        self.seven_peaks_names = ['Ru', 'Mo', 'Sm', 'Nd', 'Gd', 'Yd',
                                  'Random_class_seven']
        self.three_peak_classifier = pickle.load(
            open(path + '/' + 'lgb_model_three.pickle.dat', "rb"))
        self.four_peak_classifier = pickle.load(
            open(path + '/' + 'lgb_model_four.dat', "rb"))
        self.five_peak_classifier = pickle.load(
            open(path + '/' + 'lgb_model_five.pickle.dat', "rb"))
        self.seven_peak_classifier = pickle.load(
            open(path + '/' + 'lgb_model_seven.pickle.dat', "rb"))
        self.path = path

    def round_up_values_higher_charge(self, test):
        """
        Round m/z ratio decimals to 0, 1/4, 1/3, 1/2,2/3, 3/4, or 1.
        Return list of rounded m/z values

        Note: paper says they round but code actually takes certain ranges
        of values and assigns them a decimal amount e.g 0-0.2 = 0, 0.22-0.27=
        0.25, 0.27-0.35=0.33, 0.37-0.6=.5, 0.6-0.7=.67, 0.7, 0.8=0.75,
        0.8-1=1. This decimal amount is added to the original number converted
        to an integer.

        THIS method not only results in many many rounding errors
        but also has gaps in it causing numbers between 0.2 and 0.22 as well as
        between 0.35 and 0.37 to be assigned to 1.
        Arguments -------
        test: column of dataframe with m/z values
        """
        new_test = []
        for n in test:
            temp = int(n) # conv to int / lop off decimal
            if n <= temp + 0.2:
                new_n = temp
                new_test.append(new_n)
            elif (n > temp + 0.22 and n <= temp + 0.27):
                new_n = temp + 0.25
                new_test.append(new_n)
            elif (n > temp + 0.27 and n <= temp + 0.35):
                new_n = temp + 0.33
                new_test.append(new_n)
            elif (n > temp + 0.37 and n <= temp + 0.6):
                new_n = temp + 0.5
                new_test.append(new_n)
            elif (n > temp + 0.6 and n <= temp + 0.7):
                new_n = temp + 0.67
                new_test.append(new_n)
            elif (n > temp + 0.7 and n <= temp + 0.8):
                new_n = temp + 0.75
                new_test.append(new_n)
            else:
                new_n = temp + 1
                new_test.append(new_n)
        return new_test

    def three_peak_class(self, peak):
        """
        Takes in 3 peaks and classifies them as an element or unknown. Only
        model which doesn't use standard deviation of abundances as a feature.

        Return predictions, name of element with highest predicted probability,
        and dataframe passed to model with computed standard deviation added.
        """

        # Convert peak data into dataframe model expects
        x = {'P1': [peak['normalized_count'].iloc[0]],
             'P2': [peak['normalized_count'].iloc[1]],
             'P3': [peak['normalized_count'].iloc[2]]}
        x1 = pd.DataFrame(x)

        # Predict class of peak and get name of class
        pred = self.three_peak_classifier.predict(x1)
        name = self.three_peaks_names[np.argmax(pred[0])]

        # Compute standard deviation of peaks abundance
        x1['std'] = x1.std(axis=1)
        return pred, name, x1

    def four_peak_class(self, peak):
        """
        Takes in 4 peaks and classifies them as an element or unknown.

        Note: 4 peak model alone uses ratio of peak abundance rather than peak
        abundance alone as features for the model. Features are ratio of Peak
        1 to Peaks 2, 3, and 4 respectively as well as the ratio of Peak 2 to
        peaks 3 and 4 and finally the ratio of Peak 3 to peak 4, also uses the
        standard deviation of the 4 peaks as a feature.

        Return predictions, name of element with highest predicted probability,
        and dataframe passed to model with computed standard deviation added.
        """

        # Convert peak data to dataframe model expects
        x = {'P1': [peak['normalized_count'].iloc[0]],
             'P2': [peak['normalized_count'].iloc[1]],
             'P3': [peak['normalized_count'].iloc[2]],
             'P4': [peak['normalized_count'].iloc[3]]}
        x1 = pd.DataFrame(x)
        x1['std'] = x1.std(axis=1)
        x1['P1/P2'] = x1['P1'] / x1['P2']
        x1['P1/P3'] = x1['P1'] / x1['P3']
        x1['P1/P4'] = x1['P1'] / x1['P4']
        x1['P2/P3'] = x1['P2'] / x1['P3']
        x1['P2/P4'] = x1['P2'] / x1['P4']
        x1['P3/P4'] = x1['P3'] / x1['P4']
        x2 = x1.drop(['P1', 'P2', 'P3', 'P4'], axis=1)
        pred = self.four_peak_classifier.predict(x2)
        name = self.four_peaks_names[np.argmax(pred[0])]
        return pred, name, x1

    def five_peak_class(self, peak):
        """
        Takes in 5 peaks and classifies them as an element or unknown.

        Return predictions, name of element with highest predicted probability,
        and dataframe passed to model with computed standard deviation added.
        """
        x = {'P1': [peak['normalized_count'].iloc[0]],
             'P2': [peak['normalized_count'].iloc[1]],
             'P3': [peak['normalized_count'].iloc[2]],
             'P4': [peak['normalized_count'].iloc[3]],
             'P5': [peak['normalized_count'].iloc[4]]}
        x1 = pd.DataFrame(x)
        x1['std'] = x1.std(axis=1)
        pred = self.five_peak_classifier.predict(x1)
        name = self.five_peaks_names[np.argmax(pred[0])]
        return pred, name, x1

    def seven_peak_class(self, peak):
        """
        Takes in 7 peaks and classifies them as an element or unknown.

        Return predictions, name of element with highest predicted probability,
        and dataframe passed to model with computed standard deviation added.
        """
        x = {'P1': [peak['normalized_count'].iloc[0]],
             'P2': [peak['normalized_count'].iloc[1]],
             'P3': [peak['normalized_count'].iloc[2]],
             'P4': [peak['normalized_count'].iloc[3]],
             'P5': [peak['normalized_count'].iloc[4]],
             'P6': [peak['normalized_count'].iloc[5]],
             'P7': [peak['normalized_count'].iloc[6]]}
        x1 = pd.DataFrame(x)
        x1['std'] = x1.std(axis=1)
        pred = self.seven_peak_classifier.predict(x1)
        name = self.seven_peaks_names[np.argmax(pred[0])]
        return pred, name, x1

    def MZ_ratio_match(self, frame, new_test):
        """
        Algorithm to match a set of peaks to a known Inter-Peak Distance Ratio.
        Takes a dataframe of all peaks and intensities,

        Returns 3 lists, one of abundance
        Arguments ------
        frame: rounded (m/z, intensity) pairs for peak finder peaks
        new_test: rounded m/z values only
        """

        # Define path to database CSVs and datastructures
        path = self.path + '/' + 'atomic_database'
        database = []
        database_std = []
        database_name = []

        # For each element read in its abundance pattern, if enough of it is
        # present in this file add it to the database.
        for filename in os.listdir(path):
            temp = pd.read_csv(path + '/' + filename)
            m = temp['m'].isin(new_test)  # get masses from element in spectrum
            cols = temp.index[m].tolist()
            comp = temp['Composition'].iloc[cols].tolist() # get abundances

            # If abundances sum above 99 and there are 3 or more peaks found
            # and there are more than 3 abundances (? len(comp) should equal
            # len(cols) ?) then add to database.
            if (len(cols) >= 3 and sum(comp) >= 99 and len(comp) >= 3):
                # get rows of dataframe relevant to current element
                x = frame[frame['Da'].isin(temp['m'][m])]
                # convert count to abundance
                normalized_I = x['count'] / sum(x['count']) * 100
                x['normalized_count'] = normalized_I
                # get standard deviation of full known abundance
                std_comp = temp['Composition'].std()
                std_temp = x['normalized_count'].std()
                database.append(x)
                database_std.append(std_comp)
                fn = filename.split('.')
                database_name.append(fn[0])
        return database, database_std, database_name

    def multi_peak_search(self, frame, new_test):
        """
        Searches a dataframe of rounded peaks and their intensities for
        matches with known elements or compounds.

        Returns a tuple of lists: names of predicted elements, dataframe
        row used for prediction, predictions from that dataframe, list of name
        predicted by model and name thought possible by database, list of
        elements that model was uncertain about or had large standard deviation
        Arguments ------
        frame: rounded (m/z, intensity) pairs for peak finder peaks
        new_test: rounded m/z values only

        Note: Paper says uses 0.9 confidence from model prediction, actually
        uses 0.8 for 3 and 4 peak models. Uses no restriction on model
        confidence for 5 and 7 peak models. Only thesholds those results by
        0.5 standard deviation not mentioned in paper

        Also paper mentions searching for peaks by Inter-Peak Distance Ratio
        this does not appear to happen at all.
        """
        # Get list of elems and their relevant data possible to be in spectrum
        database, database_std, database_name = self.MZ_ratio_match(frame,
                                                                    new_test)
        name_pair = []
        predictions = []
        new_peaks = []
        name_pair_uncertain = []
        new_peaks_uncertain = []
        predictions_uncertain = []

        for name, peak, std in zip(database_name, database, database_std):
            if len(peak) == 3:
                prediction, pred_name, x = self.three_peak_class(peak)
                std_deviation = abs(x['std'][0] - std) / std
                if max(prediction[0]) <= 0.8:
                    x_uncertain = sorted(
                        zip(prediction[0], self.three_peaks_names),
                        reverse=True)[:3]
                    x__uncertain_pred = sorted(prediction[0], reverse=True)[:3]
                if std_deviation <= 0.5 and max(prediction[0]) >= 0.8:
                    name_pair.append([pred_name, name])
                    predictions.append(prediction)
                    new_peaks.append(peak)
                elif (std_deviation <= 0.5 and max(prediction[0]) <= 0.8):
                    name_pair_uncertain.append(
                        [x_uncertain[0][1], x_uncertain[1][1],
                         x_uncertain[2][1], name])
                    predictions_uncertain.append(prediction)
                    new_peaks_uncertain.append(peak)

            if len(peak) == 4:
                prediction, pred_name, x = self.four_peak_class(peak)
                std_deviation = abs(x['std'][0] - std) / std
                if max(prediction[0]) <= 0.8:
                    x_uncertain = sorted(
                        zip(prediction[0], self.four_peaks_names),
                        reverse=True)[:3]
                    x__uncertain_pred = sorted(prediction[0], reverse=True)[:3]
                if (std_deviation <= 0.5 and max(prediction[0]) >= 0.8):
                    name_pair.append([pred_name, name])
                    predictions.append(prediction)
                    new_peaks.append(peak)
                elif (std_deviation <= 0.5 and max(prediction[0]) <= 0.5):
                    name_pair_uncertain.append(
                        [x_uncertain[0][1], x_uncertain[1][1],
                         x_uncertain[2][1], name])
                    predictions_uncertain.append(prediction)
                    new_peaks_uncertain.append(peak)

            if len(peak) == 5:
                prediction, pred_name, x = self.five_peak_class(peak)
                std_deviation = abs(x['std'][0] - std) / std
                if std_deviation <= 0.5:
                    name_pair.append([pred_name, name])
                    predictions.append(prediction)
                    new_peaks.append(peak)

            if len(peak) == 7:
                prediction, pred_name, x = self.seven_peak_class(peak)
                std_deviation = abs(x['std'][0] - std) / std
                if std_deviation <= 0.5:
                    name_pair.append([pred_name, name])
                    predictions.append(prediction)
                    new_peaks.append(peak)

        elements_three_or_more_peak = []
        for item in name_pair:
            if item[0] == item[1]:
                print('Element {} is confirmed!'.format(item[0]))
                elements_three_or_more_peak.append(item[1])
            if item[0] in item[1]:
                print('Element {} is confirmed!'.format(item[1]))
                elements_three_or_more_peak.append(item[1])
        if len(name_pair_uncertain) > 0:
            print('there are peaks with uncertainies!')
            all_elements_uncertain = [name_pair_uncertain, new_peaks_uncertain,
                                      predictions_uncertain]
        else:
            all_elements_uncertain = []
        return elements_three_or_more_peak, new_peaks, predictions, name_pair, all_elements_uncertain

    def two_peak_search(self, frame, new_test):
        """
        Searches dataframe of rounded m/zs and their intensities for 2 isotope
        elements.

        Matches 2 peak by std deviation of abundance ratio, uses .5 instead of
        .3 mentioned in paper.

        Returns list of confirmed 2 isotope / peak elements.
        Arguments ------
        frame: rounded (m/z, intensity) pairs for peak finder peaks
        new_test: rounded m/z values only
        """
        path = self.path + '/' + 'atomic_database'
        database = []
        database_std = []
        database_name = []
        total_count = sum(frame['count'])
        for filename in os.listdir(path):
            #    print(filename)
            temp = pd.read_csv(path + '/' + filename)
            m = temp['m'].isin(new_test)
            cols = temp.index[m].tolist()
            comp = temp['Composition'].iloc[cols].tolist()
            #    print(m)
            std_comp = temp['Composition'].std()
            x = frame[frame['Da'].isin(temp['m'][m])]
            if (len(cols) == 2 and sum(comp) >= 98 and len(x) > 0):
                print(frame[frame['Da'].isin(temp['m'][m])])
                normalized_I = x['count'] / sum(x['count']) * 100
                x['normalized_count'] = normalized_I
                if ((sum(x['count']) / total_count) >= 0):
                    std = x['normalized_count'].std()
                    std_deviation = abs(std_comp - std) / std_comp
                    if std_deviation <= 0.5:
                        database.append(x)
                        database_std.append(std_deviation)
                        fn = filename.split('.')
                        database_name.append(fn[0])
                        print(fn)
        elements_two_peak = database_name
        return elements_two_peak

    def one_peak_search(self, frame, new_test):
        """
        Searches dataframe of rounded m/zs and their intensities for 1 isotope
        elements.

        Matches 1 peak if it is in the database, is above 98% abundance, and
        makes up more than .0001 of total counts in spectrum.

        Returns list of confirmed 1 isotope / peak elements.
        Arguments ------
        frame: rounded (m/z, intensity) pairs for peak finder peaks
        new_test: rounded m/z values only
        """
        path = self.path + '/' + 'atomic_database'
        database = []
        database_std = []
        database_name = []
        total_count = sum(frame['count'])
        for filename in os.listdir(path):
            #    print(filename)
            temp = pd.read_csv(path + '/' + filename)
            m = temp['m'].isin(new_test)
            cols = temp.index[m].tolist()
            comp = temp['Composition'].iloc[cols].tolist()
            std_comp = temp['Composition'].std()
            x = frame[frame['Da'].isin(temp['m'][m])]
            if (len(cols) == 1 and sum(comp) >= 98 and len(x) > 0):
                print(comp)
                normalized_I = x['count'] / sum(x['count']) * 100
                x['normalized_count'] = normalized_I
                if ((sum(x['count']) / total_count) >= 0.0001):
                    std_deviation = 0
                    database.append(x)
                    database_std.append(std_deviation)
                    fn = filename.split('.')
                    database_name.append(fn[0])
        elements_one_peak = database_name
        return elements_one_peak

    def organic_peak_search(self, frame, new_test):
        """
        Searches dataframe of rounded m/zs and their intensities for organic
        molecules.

        Matches 3 or fewer peaks if they appear in a spectrum together and
        make up 98% of abundance.

        Returns list of confirmed organic molecules.
        Arguments ------
        frame: rounded (m/z, intensity) pairs for peak finder peaks
        new_test: rounded m/z values only
        """
        database = []
        database_std = []
        database_name = []
        path = self.path + '/' + 'organic_database'
        elements_organic_peak = []
        for filename in os.listdir(path):
            #    print(filename)
            temp = pd.read_csv(path + '/' + filename)
            m = temp['m'].isin(new_test)
            cols = temp.index[m].tolist()
            comp = temp['Composition'].iloc[cols].tolist()

            if (len(cols) <= 3 and sum(comp) >= 98):
                print(frame[frame['Da'].isin(temp['m'][m])])
                std_comp = temp['Composition'].std()
                x = frame[frame['Da'].isin(temp['m'][m])]
                normalized_I = x['count'] / sum(x['count']) * 100
                x['normalized_count'] = normalized_I
                database.append(x)
                fn = filename.split('.')
                elements_organic_peak.append(fn[0])
                print(fn)
        return elements_organic_peak


from sklearn.model_selection import train_test_split
import lightgbm as lgb


class Molecular_pattern_recognizer(object):
    def __init__(self):
        pass

    def combine_ions(self, comp_1, comp_2):
        all_ion = []
        all_compo = []
        total_S = comp_1['m']
        compo_S = comp_1['Composition']
        total_Cu = comp_2['m']
        compo_Cu = comp_2['Composition']
        for i, s in zip(compo_S, total_S):
            for j, cu in zip(compo_Cu, total_Cu):
                ion = s + cu
                compo = round(i * j / 100, 2)
                all_ion.append(ion)
                all_compo.append(compo)
        CuS = pd.DataFrame(data={'m': all_ion, 'Composition': all_compo})
        CuS = CuS.groupby('m').sum().reset_index()
        x = CuS[(CuS != 0).all(1)].reset_index(drop=True)
        return x

    def molecule_formula(self, *args):
        # print(args)
        temp = args[0]
        if args[1] >= 2:
            for _ in range(1, args[1]):
                temp = self.combine_ions(temp, args[0])
        #        new_temp = temp
        for i in range(2, len(args), 2):
            if args[i + 1] > 0:
                for _ in range(args[i + 1]):
                    temp = self.combine_ions(temp, args[i])

        final = temp[temp['Composition'] > 1]
        return final

    def molecule_all(self, *args):
        all_molecule = []
        stats = []
        for charge in range(1, 5):
            if len(args[1]) == 1:
                for n in range(1, 4):
                    for m in range(0, 4):
                        molecule = self.molecule_formula(args[0], n, args[1][0],
                                                         m)
                        molecule['m'] /= charge
                        all_molecule.append(molecule)
                        # print(molecule)
                        stat = [n, m, charge]
                        stats.append(stat)

            if len(args[1]) == 2:
                for n in range(1, 4):
                    for m in range(0, 4):
                        for a in range(0, 4):
                            molecule = self.molecule_formula(args[0], n,
                                                             args[1][0], m,
                                                             args[1][1], a)
                            molecule['m'] /= charge
                            all_molecule.append(molecule)
                            # print(molecule)
                            stat = [n, m, a, charge]
                            stats.append(stat)

            if len(args[1]) == 3:
                for n in range(1, 4):
                    for m in range(0, 4):
                        for a in range(0, 4):
                            for b in range(0, 4):
                                molecule = self.molecule_formula(args[0], n,
                                                                 args[1][0], m,
                                                                 args[1][1], a,
                                                                 args[1][2], b)
                                molecule['m'] /= charge
                                all_molecule.append(molecule)
                                # print(molecule)
                                stat = [n, m, a, b, charge]
                                stats.append(stat)
        stats = pd.DataFrame(stats)
        return all_molecule, stats

    def find_good_molecules(self, frame, metals, metals_names, nonmetals,
                            nonmetals_names):
        all_molecule = []
        stats = []
        N_metal = []
        total = []
        good_molecules = []
        metals_dev = []
        normalized_Is = []
        MZ_ratio_round = frame['Da'].tolist()
        for N, metal in enumerate(metals):
            all_molecule, stats = self.molecule_all(metal, nonmetals)
            for i, molecule in enumerate(all_molecule):
                m = molecule['m'].isin(MZ_ratio_round).tolist()
                index = np.where(m)[0]
                comp_molecule = np.array(molecule['Composition'])[index]
                x = frame[frame['Da'].isin(molecule['m'][m])]
                normalized_I = x['count'] / sum(x['count']) * 100
                x['normalized_count'] = normalized_I
                deviation = sum(abs(normalized_I - comp_molecule))

                if (sum(comp_molecule) >= 98 and len(
                        comp_molecule) > 0 and deviation <= 20):
                    if (sum(stats.iloc[i][1:]) <= 10 and sum(
                            stats.iloc[i][1:]) >= 2):
                        total.append(stats.iloc[i])
                        metals_dev.append(deviation)
                        N_metal.append(N)
                        good_molecules.append(molecule)
                        normalized_Is.append(x)

        total = pd.DataFrame(total)
        total['AME'] = metals_dev
        # total['N_metal'] = N_metal
        total.columns.values[0] = 'metal atoms'
        total.columns.values[len(nonmetals_names) + 1] = 'charges'
        Name_metals = []
        for i, name in enumerate(nonmetals_names):
            total.columns.values[i + 1] = name
        for j in N_metal:
            Name_metals.append(metals_names[j])

        total['metal_names'] = Name_metals

        return good_molecules, total

    def molecular_classifier(self, total):
        item = total[0]
        frame = []
        N = 1000
        comps = []
        delta = 0.01
        for n, item in enumerate(total):
            augmented_class = []
            for i in range(N):
                comps = []
                comp = item['Composition'].tolist()
                for j in range(len(item)):
                    mu, sigma = comp[j], delta * comp[j]
                    s = np.random.normal(mu, sigma, 1)
                    comps.append(s[0])
                    total = sum(comps)
                    if (total <= (sum(comp) + 1) and total >= (sum(comp) - 1)):
                        augmented_class.append(comps)
            augmented_class = pd.DataFrame(augmented_class)
            augmented_class['Class'] = n
            frame.append(augmented_class)
        data = pd.concat(frame)
        X_data = data.drop('Class', axis=1)
        y = data.Class

        print('training test spilting...')
        seed = 7
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X_data, y,
                                                            test_size=test_size,
                                                            random_state=seed)
        print('start_training...')
        evals_result = {}
        params = {
            "objective": "multiclass",
            "num_class": len(frame),
            "num_leaves": 30,
            "max_depth": 5,
            "learning_rate": 0.01,
            "bagging_fraction": 0.9,  # subsample
            "feature_fraction": 0.9,  # colsample_bytree
            "bagging_freq": 5,  # subsample_freq
            "bagging_seed": 20,
            "verbosity": -1,
            'metric': 'multi_logloss'}
        lgtrain, lgval = lgb.Dataset(X_train, y_train), lgb.Dataset(X_test,
                                                                    y_test)

        model_lgb = lgb.train(params, lgtrain, 2000,
                              valid_sets=[lgtrain, lgval],
                              evals_result=evals_result,
                              early_stopping_rounds=100, verbose_eval=200)
        # print('Plotting metrics recorded during training...')
        # lgb.plot_metric(evals_result, metric='multi_logloss')
        # plt.xlim(0,1000)
        # plt.show()
        return model_lgb
