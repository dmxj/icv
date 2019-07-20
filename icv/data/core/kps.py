# -*- coding: UTF-8 -*-
import numpy as np

class Keypoint(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def x_int(self):
        """
        Return the keypoint's x-coordinate, rounded to the closest integer.
        Returns
        -------
        result : int
            Keypoint's x-coordinate, rounded to the closest integer.
        """
        return int(np.round(self.x))

    @property
    def y_int(self):
        """
        Return the keypoint's y-coordinate, rounded to the closest integer.
        Returns
        -------
        result : int
            Keypoint's y-coordinate, rounded to the closest integer.
        """
        return int(np.round(self.y))

    def copy(self, x=None, y=None):
        """
        Create a shallow copy of the Keypoint object.
        Parameters
        ----------
        x : None or number, optional
            Coordinate of the keypoint on the x axis.
            If ``None``, the instance's value will be copied.
        y : None or number, optional
            Coordinate of the keypoint on the y axis.
            If ``None``, the instance's value will be copied.
        Returns
        -------
        imgaug.Keypoint
            Shallow copy.
        """
        return self.deepcopy(x=x, y=y)

    def deepcopy(self, x=None, y=None):
        """
        Create a deep copy of the Keypoint object.
        Parameters
        ----------
        x : None or number, optional
            Coordinate of the keypoint on the x axis.
            If ``None``, the instance's value will be copied.
        y : None or number, optional
            Coordinate of the keypoint on the y axis.
            If ``None``, the instance's value will be copied.
        Returns
        -------
        imgaug.Keypoint
            Deep copy.
        """
        x = self.x if x is None else x
        y = self.y if y is None else y
        return Keypoint(x=x, y=y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "Keypoint(x=%.8f, y=%.8f)" % (self.x, self.y)
