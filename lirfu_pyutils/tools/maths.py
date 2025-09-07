from typing import List, Optional, Any
from math import prod

import numpy as np


def lerp(t:Any, a:Any, b:Any):
    """
        Linearly interpolate values [a,b] using parameter t.
    """
    return a + (b - a) * t

def normalize_range(values:Any, eps:float=1e-12) -> Any:
    """
        Returns the values normalized to the [0,1] interval.
    """
    m = values.min()
    return (values - m) / max((values.max() - m), eps)


def gaussian_1d_kernel(std:float, kernel_size:Optional[int]=None, trunc_stds:int=4) -> np.ndarray:
    """
        Make a 1D Gaussian kernel.

        Parameters
        ----------
        std : float
            Standard deviation of the Gaussian.
        kernel_size : int, optional
            Size of the convolutional kernel. If `None` is calculated by: `max(3, int(std*trunc_stds+0.5))`. Default: None.
        trunc_stds : int
            Number of standard deviations to truncate at. Default: 4.
    """
    if kernel_size is None:
        kernel_size = max(3, int(std * trunc_stds + 0.5))  # Kernel cropped at 4 standard deviations.
    k = np.linspace(-kernel_size/2, kernel_size/2, kernel_size)
    k = np.exp( -0.5 / std**2 * k**2 )
    return k

def gaussian_2d_kernel(std:float, kernel_size:Optional[int]=None, trunc_stds:int=4) -> np.ndarray:
    """
        Make a 2D Gaussian kernel.

        Parameters
        ----------
        std : float
            Standard deviation of the Gaussian.
        kernel_size : int, optional
            Size of the convolutional kernel. If `None` is calculated by: `max(3, int(std*trunc_stds+0.5))`. Default: None.
        trunc_stds : int
            Number of standard deviations to truncate at. Default: 4.
    """
    k = gaussian_1d_kernel(std, kernel_size, trunc_stds)[None,:]
    k = k * k.T
    k /= k.sum()
    return k


def round_down(number:int, target:int) -> int:
    """
        Rounds number to largest fitting multiple of target value.
    """
    return (number // target) * target

def primes(n: int) -> List[int]:
    """
        Find primes using Sieve of Erathostenes.
    """
    if n <= 2: return None
    sieve = [True,] * (n+1)
    for x in range(3, int(n**0.5)+1, 2):
        for y in range(3, (n//x)+1, 2):
            sieve[(x*y)] = False
    return [2]+[i for i in range(3,n,2) if sieve[i]]

def count_divisions_per_prime(n: int, primes: List[int]) -> List[int]:
    """
        Count number of divisions of each prime factor. Removes unused primes from the original list.
    """
    ctr = [0,]*len(primes)
    i = 0
    for j in range(len(primes)):
        j = j - i
        p = primes[j]
        while n % p == 0:
            n = n // p
            ctr[j] += 1
        if ctr[j] == 0:
            ctr.pop(j)
            primes.pop(j)
            i += 1
    return ctr

def generate_alternating_sequence(i: int) -> int:
    """
        Used for generation of the addition factors that, when accumulated, generate the sequence: [+1, -1, +2, -2, +3, ...].
        Returns next addition factor from sequence: [+1, -2, +3, -4, +5, ...].

        Parameters
        ----------
        i: int
            Current addition factor.
        Returns
        ----------
        i: int
            Next addition factor in the sequence.
    """
    return -i - i // abs(i)

def alternating_fix_primes(self, n):
    primes = self.__primes(n)
    divs = self.__count_divisions(n, primes)
    i = primes.index(self.value)
    d = divs[i]
    leftover = self.num - d
    offset = +1  # +1,-1,+2,-2,...
    seq = +1  # +1,-2,+3,-4,...
    try:
        for _ in range(leftover):
            while (i+offset) < 0 or divs[i+offset] == 0:  # Find next closest prime.
                seq = self.__alternating_sequence(seq)
                offset += seq
            # Move from this prime to next.
            divs[i+offset] -= 1
            divs[i] += 1
        return prod([v**m for m,v in zip(divs,primes)])
    except IndexError as e:
        raise RuntimeError(f'Number {n} is not divisible by {self.value}**{self.num}! primes={primes}, ctr={divs}')