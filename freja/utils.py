def golden_section(f, a, b, tol=1e-5, **kwargs):
    """
    Golden section search.

    Stolen from https://en.wikipedia.org/wiki/Golden-section_search

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    example:
    >>> f = lambda x: (x-2)**2
    >>> a = 1
    >>> b = 5
    >>> tol = 1e-5
    >>> (c,d) = gss(f, a, b, tol)
    >>> print (c,d)
    (1.9999959837979107, 2.0000050911830893)
    """
    import numpy as np

    # 1/phi
    invphi = (np.sqrt(5) - 1) / 2
    # 1/phi^2
    invphi2 = (3 - np.sqrt(5)) / 2

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return (a, b)

    # required steps to achieve tolerance
    n = int(np.ceil(np.log(tol / h) / np.log(invphi)))

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f(c, **kwargs)
    yd = f(d, **kwargs)

    for k in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f(c, **kwargs)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d, **kwargs)

    if yc < yd:
        return (a, d)
    else:
        return (c, b)
