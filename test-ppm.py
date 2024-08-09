# test-ppm.py
# (c) 2024-06-28, Philip D Loewen

from PrettyPrinter import ppm
import numpy as np

A0 = 2.*np.random.rand(3,3) - np.ones((3,3))

print(" ")

A = A0/10
ppm(A,"A0 / 10")

print(" ")

ppm(A0,"A0")

print(" ")


A10 = A0*10
ppm(A10,"A10 = A0 * 10")

print(" ")

A100 = A0*100
ppm(A100,"A100 = A0 * 100")

print(" ")

A1000 = A0*1000
ppm(A1000,"A1000 = A0 * 1000")

print(" ")

A10000 = A0*10000
ppm(A10000,"A10000 = A0 * 10000")

print(" ")

Abig = A0*1E23
ppm(Abig,"Abig")


print(" ")

Atiny = A0*1E-11
ppm(Atiny,"Atiny")


print(" ")

B = np.array([1,2,3,4,5]) - 0.6
ppm(B,"B")

print(" ")

D = np.random.rand(3,3,3)
ppm(D,"D")

print(" ")

C = np.pi
ppm(C,"C")

print(" ")

A10000 = A0*10000
ppm(A10000,"A10000 = A0 * 10000",sigfigs=6)

print(" ")

ppm(A10000,"A10000 = A0 * 10000",sigfigs=5)
print(" ")

ppm(A10000,"A10000 = A0 * 10000",sigfigs=4)
print(" ")

ppm(A10000,"A10000 = A0 * 10000",sigfigs=3)
print(" ")

ppm(A10000,"A10000 = A0 * 10000",sigfigs=2)
print(" ")

ppm(A10000,"A10000 = A0 * 10000",sigfigs=1)
print(" ")

ppm(A10000,"A10000 = A0 * 10000",sigfigs=0)
print(" ")

ppm(A10000,"A10000 = A0 * 10000",sigfigs=48)
print(" ")

yyy = np.random.rand(1,7,3)*1E7
ppm(yyy,"yyy")
ppm(yyy[:,:,2],"yyy[:,:,2]")

print(" ")

zzz = np.random.rand(3,1,4)*1E-5
ppm(zzz,"zzz")
ppm(zzz[:,:,2],"zzz[:,:,2]")

print(" ")

# It turns out that a 0-dimensional array is possible,
# and it has ndim=0, size=1, and reshaping it just works.
# The following demonstrates this.

scalar = np.array(666.)
ppm(scalar,"scalar")


