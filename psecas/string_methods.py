def var_replace(eq, var, new):
    """
    Replace all instances of string var with string new.
    This function differs from the default string replace method in
    that it only makes the replace if var is not contained inside a
    word.

    Example:
    eq = "-1j*kx*v*drho -drhodz*dvz -1.0*dz(dvz) - drho"
    var_replace(eq, 'drho', 'foo')
    returns '-1j*kx*v*foo -drhodz*dvz -1.0*dz(dvz) - foo'
    where drhodz has not been replaced.
    """
    pos = 0
    while pos != -1:
        pos = eq.find(var, pos)
        if pos != -1:
            substitute = True
            # Check if character to the left is a letter
            if pos > 0:
                if eq[pos - 1].isalpha():
                    substitute = False
            # Check if character to the right is a letter
            if pos + len(var) < len(eq):
                if eq[pos + len(var)].isalpha():
                    substitute = False
            if substitute:
                eq = eq[:pos] + new + eq[pos + len(var) :]
            # Increment pos to prevent the function from repeatedly
            # finding the same occurrence of var
            else:
                pos += len(var)
    return eq