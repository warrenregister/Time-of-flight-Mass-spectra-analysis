"""
Class to handle calculating and storing IDRs for various elements.
"""
from functools import reduce
from math import gcd


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
