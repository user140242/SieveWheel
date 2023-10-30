# SieveWheel
Optimize the code for a bit segmented sieve wheel

This repository concerns a refinement of the sieve of Eratosthenes.

The traditional sieve of Eratosthenes described on this [Wikipedia page](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) has been modified through the use of modular arithmetic, equivalent to the use of the wheel factorization, as described on this [Wikipedia page](https://en.wikipedia.org/wiki/Wheel_factorization), so as not to consider all the multiples of the prime numbers chosen as the basis and in this way reduce the memory used.

![SieveWheel](https://github.com/user140242/SieveWheel/assets/108657671/bdab580e-77df-4a9c-8e44-0a30fb12f1f8)

To obtain prime numbers less than N, once the wheel size (equal to the modulus) is chosen a bitvector of size (N/modulus)*phi(modulus) is used, with phi() Euler's totient function and phi(modulus) corresponding to the number of residue classes used.


![Wheel30](https://github.com/user140242/SieveWheel/assets/108657671/a0519eae-4f66-4d6a-ba17-04f020c121fe)

For higher values of N a segmented version is carried out and the size of the segment is given by the product of the prime numbers following those of the base in a number equal to the value of *p_mask_i* variable. This is to carry out further optimization by initializing the segment in order to eliminate multiples of the numbers chosen using *p_mask_i*. For example in the case modulus=210 and *p_mask_i*=4 segments of size equal to 11 \*13 \*17 \*19 and the number of residue classes is phi(210)=48 therefore using 6 bytes to store the 48 residue classes, a bitvector of size 11\*13\*17\*19\*6 = 271 kB is used for each segment.

A file with a demo version is published. I'm not a programming expert. The purpose of this repository is to help you make an optimized version that is easy to use via command line, once compiled, and if possible get a faster version.


