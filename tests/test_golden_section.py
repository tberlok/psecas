def test_golden_section():
    """Test golden section method on a simple example"""
    import numpy as np
    from psecas import golden_section

    def f(x):
        return (x-2)**2

    tol = 1e-8
    (c, d) = golden_section(f, 1, 5, tol)
    print(c, d)
    np.testing.assert_allclose(c, 2.0, tol)
    np.testing.assert_allclose(d, 2.0, tol)


if __name__ == '__main__':
    test_golden_section()
