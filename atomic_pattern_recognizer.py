"""
Class for identifying location of elemental isotope peaks in a TOF Spectra.
"""
import itertools

import pandas as pd
import numpy as np
import pickle
from os import listdir
from math import gcd
from functools import reduce
from math import gcd
from functools import reduce


class InterPeakDistanceRatio:
    """
    Class to handle calculating and storing IDRs for various elements.
    """

    def __init__(self, nums):
        """
        Converts list of mass values to inter-peak distance ratio, divides
        by greatest common divisor to simplify ratio.
        """
        nums = list(nums)
        nums = sorted(nums)
        length = len(nums)
        self.idr = ''
        dists = []

        i = 0
        while 1:
            if i + 1 < length:
                distance = abs(nums[i] - nums[i + 1])
                dists.append(distance)
                i += 1
            else:
                break
        if min(dists) == .5:
            dists = [2 * x for x in dists]
        dists = [int(x) for x in dists]
        divisor = reduce(gcd, dists)
        dists = [x / divisor for x in dists]

        for i, num in enumerate(dists):
            self.idr += str(int(num))
            if i < len(dists) - 1:
                self.idr += ':'

    def __repr__(self):
        """
        Returns string representation of IDR
        """
        return self.idr

    def __eq__(self, other):
        """
        Returns true if object's IDR is the same.
        """
        return self.idr == other.idr

    def get_dists(self):
        """
        Return distance ratio as a list of integers.
        """
        dists = [int(float(x)) for x in self.idr.split(':')]
        return dists


class AtomicPatternRecognizer:
    def __init__(self, path):
        """
        Initiate APR with models used for multiple peak identification and
        names of all elements being searched for.

        Arguments -------
        path: path to folder with pickled models in it
        """
        self.three_peaks_names = ['Mg', 'S', 'Si', 'Fe', 'Ca_3P',
                                  'double_S',
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
        self.idrs = self.get_idrs()

    @staticmethod
    def round_mz_ratio(masses):
        """
        Round decimal of m/z numbers to nearest value in 0, 1/4, 1/3, 1/5, 2/3,
        3/4, 1

        Arguments -------
        masses: iterable of mass values for a spectra
        """
        rounded_masses = []
        nums = np.array([0, 1 / 4, .33, 1 / 2, .67, 3 / 4, 1])
        for mass in masses:
            integer = int(mass)
            differences = abs(nums - (mass - integer))
            ind = np.where(differences == min(differences))
            rounded_masses.append(float(integer + nums[ind]))

        return rounded_masses

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

    def get_idrs(self):
        """
        Determines all possible inter-peak distance ratios present in elemental
        database.
        """
        path = self.path + 'atomic_database'
        idrs = []
        for file in listdir(path):
            csv = pd.read_csv(path + '/' + file)
            if len(csv['m']) >= 3:
                idr = InterPeakDistanceRatio(csv['m'])
                if idr not in idrs:
                    idrs.append(idr)
        return idrs

    def idr_search(self, frame, mass_col='masses'):
        """
        Search set of peaks for element isotope candidates by looking at
        inter-peak distance ratio.

        Returns a list of tuples containing the indices of a set of peaks
        in frame which have the same IDR as a known isotope.
        """
        candidates = []
        for num in range(0, len(frame), 3):
            if num + 15 >= len(frame):
                num = len(frame) - 16
            for ratio in [3, 4, 5, 7]:
                for comb in itertools.combinations(
                        list(range(num, num + 15)), ratio):
                    nums = []
                    for i in comb:
                        nums.append(frame[mass_col].loc[i])
                    idr = InterPeakDistanceRatio(nums)
                    if idr in self.idrs:
                        candidates.append(comb)
        return candidates

    def multi_peak_search_idr(self, frame, mass_col='masses',
                          inten_col='intensities'):
        """
        Checks if candidate isotopes combinations are actually isotopes by
        feeding to machine learning algorithms to check.
        """
        candidates = self.idr_search(frame, mass_col)
        names = []
        predictions = []
        peaks = []

        uncertain_names = []
        uncertain_preds = []
        uncertain_peaks = []

        for tup in candidates:
            cand = pd.DataFrame(columns=['normalized_count'])
            sum = 0
            for i, ind in enumerate(tup):
                sum += frame.loc[ind][inten_col]
                cand.loc[i] = {'normalized_count': frame.loc[ind][inten_col]}
            cand['normalized_count'] = cand['normalized_count'] / sum

            length = len(cand)
            pred = [0]
            name = ''
            if length == 3:
                pred, name, features = self.three_peak_class(cand)
            elif length == 4:
                pred, name, features = self.four_peak_class(cand)
            elif length == 5:
                pred, name, features = self.five_peak_class(cand)
            elif length == 7:
                pred, name, features = self.seven_peak_class(cand)

            if max(pred) > 0.8:
                names.append(name)
                predictions.append(pred)
                peaks.append(cand)
            elif len(pred) > 1:
                uncertain_names.append(name)
                uncertain_preds.append(pred)
                uncertain_peaks.append(cand)

            return (names, predictions, peaks, uncertain_names, uncertain_preds,
                    uncertain_peaks)





    def mz_ratio_match(self, frame, mass_col='masses', inten_col='intensities'):
        """
        Algorithm to match a set of peaks to a known Inter-Peak Distance Ratio.
        Takes a dataframe of all peaks and intensities,

        Returns 3 lists, one of abundance
        Arguments ------
        frame: rounded (m/z, intensity) pairs for peak finder peaks
        mass_col: column name for mass column, default 'masses'
        inten_col: column name for intensity column, default 'intensities'
        """
        # Define path to database CSVs and datastructures
        path = self.path + 'atomic_database'
        database = []
        database_std = []
        database_name = []

        # For each element read in its abundance pattern, if enough of it is
        # present in this file add it to the database.
        for filename in listdir(path):
            temp = pd.read_csv(path + '/' + filename)
            m = temp['m'].isin(frame[mass_col])
            cols = temp.index[m].tolist()
            comp = temp['Composition'].iloc[cols].tolist()  # get abundances
