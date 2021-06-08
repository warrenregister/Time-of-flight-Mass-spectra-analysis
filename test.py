"""
Main file for testing atomic_pattern_recognizer objects
"""
from atom_pattern_recognizer_no_ml import AtomicPatternRecognizerNoML
from data_creation import read_csvs, generate_calibrated_data


def main():
    test = AtomicPatternRecognizerNoML("./atomic_database/Elements.txt")
    test_data = read_csvs('./test_spectra/')
    test_data = generate_calibrated_data(test_data)
    test_data['masses'] = test_data['masses'].apply(list)

    row = test_data.loc[1]
    elements = test.find_atomic_patterns(row['masses'], row['intensities'],
                                         thresh=.03)
    print(elements)


if __name__ == '__main__':
    main()