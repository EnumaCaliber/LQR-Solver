# PrettyPrinter

# (c) Philip D Loewen, 2024-06-28

import numpy as np

def ppm(M, name="",**kwargs):
    """ppm - Pretty-print a numpy ndarray, provided ndim is not too large.

       (One could say more here.)
    """
    # Main keyword argument is "sigfigs".

    if len(name)==0:
        padname=" "
        parname=" "
    else:
        padname=" "+name+" "
        parname=", namely "+name+", "

    if not isinstance(M,np.ndarray):
        print(f"Error: ppm works on objects with class numpy.ndarray, ")
        print(f"but the given object{parname}has class {M.__class__.__name__}.")
        print("No output.")
        return(None)

    if M.ndim == 1:
        MMM = M.reshape(M.shape[0],1)
        print(f"Presenting 1D array{padname}in shape {M.shape[0]}x1",end="")        
    elif M.ndim == 2:
        MMM = M
        print(f"Matrix{padname}has shape {M.shape[0]}x{M.shape[1]}",end="")
    elif M.ndim == 3 and min(M.shape) == 1:
        if M.shape[1]==1:
            MMM = M.reshape(M.shape[0],M.shape[2])
            print(f"Presenting 3D array{padname}as a lineup of {M.shape[2]} columns, ",end="")
            print(f"each with {M.shape[0]} rows",end="")
        elif M.shape[0]==1:
            MMM = M.T.reshape(M.shape[2],M.shape[1])
            print(f"Presenting 3D array{padname}as a pile of {M.shape[2]} rows, ",end="")
            print(f"each with {M.shape[1]} cols",end="")
        else:
            MMM = M.reshape(M.shape[0],M.shape[1])
            print(f"Presenting 3D array{padname}as a single {M.shape[0]}x{M.shape[1]} matrix",end="")
    else:
        MMM = M.reshape(M.size,1)
        print(f"Warning: Input array{padname}is {M.ndim}-dimensional, with shape {M.shape}. ",end="")
        print(f"Printing it as one column with {M.size} rows",end="")
    (rowcount,colcount) = MMM.shape

    s = int(kwargs.get('sigfigs',4))   # Default number of sig figs is hard-coded here
    s = max([1,s])                     # Silently repair bogus request

    # Preferred format has "r" digits before dot and "d" digits after,
    # using the largest absolute value of the numbers in the given matrix.
    # Of course, the sig figs "s" should nominally equal r+d.
    reference = np.max(np.abs(M))
    if reference > 0.0:
        r = 1 + int(np.floor(np.log10(reference)))
    else:
        r = 1
    d = s - r

    relax = False  # Decide whether to allow an extra char for simple decimals

    if r>0 and d>0:
        # Nominal case worked. Formulate format code and print
        print(f":")
        fmt = "{"+f":{s+2:d}.{d:d}f"+"}"
        for r in range(rowcount):
            print(" "+" ".join([fmt.format(q) for q in MMM[r,:]]))
    elif r==0 and relax:
        # Stretch case. All sig figs follow "0.".
        # Just allow an extra character column for cosmetic reasons.
        print(f":")
        fmt = "{"+f":{s+3:d}.{d:d}f"+"}"
        for r in range(rowcount):
            print(" "+" ".join([fmt.format(q) for q in MMM[r,:]]))
    else:
        # Scale the whole matrix so at least one entry shows all sig figs
        p = r - 1
        print(f";")
        print(f" ... multiply each value below by 1E{p}:")
        fmt = "{"+f":{s+2:d}.{s-1:d}f"+"}"
        for r in range(rowcount):
            print(" "+" ".join([fmt.format(q) for q in MMM[r,:]/10**p]))

    return(None)
