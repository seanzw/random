import sympy

def mod_inv(a, p):
    """Compute the modular inverse of a modulo p using Extended Euclidean Algorithm.

    Args:
        a (int): The number to find the inverse of.
        p (int): The modulus.

    Returns:
        int: The modular inverse of a modulo p.
    """
    r0, x0, y0 = p, 1, 0
    r1, x1, y1 = a, 0, 1
    if p == 1:
        return 0
    while r1 > 1:
        q = r0 // r1
        r1, r0 = r0 - q * r1, r1
        x1, x0 = x0 - q * x1, x1
        y1, y0 = y0 - q * y1, y1
    if y1 < 0:
        y1 += p
    return y1
    
def find_primitive_root(n, p):
    """Find a primitive n-th root of unity modulo p.

    Args:
        n (int): The order of the root of unity.
        p (int): A prime modulus.
    Returns:
        int: A primitive n-th root of unity modulo p.
    """
    order = p - 1
    factors = []
    d = 2
    while d * d <= order:
        if order % d == 0:
            factors.append(d)
            while order % d == 0:
                order //= d
        d += 1
    if order > 1:
        factors.append(order)
    def is_primitive_root(g, n, p):
        for factor in factors:
            if pow(g, (p - 1) // factor, p) == 1:
                return False
        return True

    for g in range(2, p):
        if is_primitive_root(g, n, p):
            if pow(g, (p - 1) // n, p) != 1:
                return pow(g, (p - 1) // n, p)
    return None

def gen_twiddles(n, root, p):
    """Generate twiddle factors for NTT.

    Args:
        n (int): Size of the transform (must be a power of two).
        root (int): A primitive n-th root of unity modulo p.
        p (int): A prime modulus.

    Returns:
        list of int: The twiddle factors.
    """
    twiddles = [1] * n
    for i in range(1, n):
        twiddles[i] = (twiddles[i - 1] * root) % p
    return twiddles

def ntt_naive(a, p, twiddles):
    """Compute the Number Theoretic Transform (NTT) of a polynomial a
    using the naive O(n^2) algorithm.

    Args:
        a (list of int): Coefficients of the polynomial to transform.
        p (int): A prime modulus.
        twiddles (list of int): Precomputed twiddle factors.

    Returns:
        list of int: The NTT of the input polynomial.
    """
    n = len(a)
    A = [0] * n
    for k in range(n):
        for j in range(n):
            A[k] = (A[k] + a[j] * twiddles[(j * k) % n]) % p
    return A

def intt_naive(A, p, inv_twiddles):
    """Compute the Inverse Number Theoretic Transform (INTT) of a polynomial A
    using the naive O(n^2) algorithm.

    Args:
        A (list of int): Coefficients of the polynomial in NTT domain.
        p (int): A prime modulus.
        inv_twiddles (list of int): Precomputed inverse twiddle factors.

    Returns:
        list of int: The INTT of the input polynomial.
    """
    n = len(A)
    inv_n = mod_inv(n, p)
    a = ntt_naive(A, p, inv_twiddles)
    for k in range(n):
        a[k] = (a[k] * inv_n) % p
    return a

def ntt_butterfly(a, p, twiddles):
    """Compute the Number Theoretic Transform (NTT) of a polynomial a
    using the Cooley-Tukey butterfly algorithm.

    Args:
        a (list of int): Coefficients of the polynomial to transform.
        p (int): A prime modulus.
        twiddles (list of int): Precomputed twiddle factors.
    
    Returns:
        list of int: The NTT of the input polynomial.
    """

    def bit_reverse_copy(a):
        n = len(a)
        result = a.copy()
        rev_i = 0
        for i in range(1, n):
            # Do the bit-reversal addition.
            bit = n >> 1
            while rev_i & bit:
                # We have a 1 in this bit position for carry.
                rev_i ^= bit
                bit >>= 1
            rev_i |= bit
            if rev_i > i:
                result[i], result[rev_i] = result[rev_i], result[i]
        return result
    
    n = len(a)
    A = bit_reverse_copy(a)
    length = 2
    while length <= n:
        half = length // 2
        step = n // length
        for i in range(0, n, length):
            for j in range(half):
                u = A[i + j]
                v = (A[i + j + half] * twiddles[j * step]) % p
                A[i + j] = (u + v) % p
                A[i + j + half] = (u - v + p) % p
        length *= 2
    return A

def intt_butterfly(A, p, inv_twiddles):
    """Compute the Inverse Number Theoretic Transform (INTT) of a polynomial A
    using the Cooley-Tukey butterfly algorithm.

    Args:
        A (list of int): Coefficients of the polynomial in NTT domain.
        p (int): A prime modulus.
        inv_twiddles (list of int): Precomputed inverse twiddle factors.
    """
    n = len(A)
    inv_n = mod_inv(n, p)
    a = ntt_butterfly(A, p, inv_twiddles)
    for k in range(n):
        a[k] = (a[k] * inv_n) % p
    return a

def poly_mult_ntt(a, b, p, root, twiddles, inv_root, inv_twiddles, ntt_impl, intt_impl):
    """Multiply two polynomials a and b using NTT with naive O(n^2) algorithm.

    Args:
        a (list of int): Coefficients of the first polynomial.
        b (list of int): Coefficients of the second polynomial.
        p (int): A prime modulus.
    Returns:
        list of int: Coefficients of the product polynomial.
    """
    n = len(a)

    A = ntt_impl(a, p, twiddles)
    B = ntt_impl(b, p, twiddles)

    # print("NTT of a:", A)
    # print("NTT of b:", B)

    C = [(A[i] * B[i]) % p for i in range(n)]

    c = intt_impl(C, p, inv_twiddles)

    return c

def poly_mult_direct(a, b, p):
    """Multiply two polynomials a and b directly in O(n^2) time.

    Args:
        a (list of int): Coefficients of the first polynomial.
        b (list of int): Coefficients of the second polynomial.
        p (int): A prime modulus.

    Returns:
        list of int: Coefficients of the product polynomial.
    """
    n = len(a)
    deg = n // 2
    c = [0] * n 
    for i in range(deg):
        for j in range(deg):
            c[i + j] = (c[i + j] + a[i] * b[j]) % p
    return c

def select_prime(deg):
    """Select a prime p such that (p - 1) is divisible by (deg * 2).

    Args:
        deg (int): Degree of the polynomials.

    Returns:
        int: A suitable prime number.
    """

    n = deg * 2
    candidate = n + 1
    while True:
        p = sympy.nextprime(candidate)
        if (p - 1) % n == 0:
            return p
        candidate = p + 1

def bench_poly_mult():
    import random
    import time

    deg = 1024  # Degree of polynomials
    # deg = 4  # Degree of polynomials
    p = select_prime(deg)
    assert((p - 1) % (deg * 2) == 0)
    assert(sympy.isprime(p))
    n = deg * 2

    a = [random.randint(0, p - 1) for _ in range(deg)] + [0] * deg
    b = [random.randint(0, p - 1) for _ in range(deg)] + [0] * deg

    print(f"degree: {deg}, n: {n}, prime p: {p}")
    if deg <= 16:
        print("Polynomial A:", a)
        print("Polynomial B:", b)

    # Benchmark direct multiplication for comparison
    start = time.time()
    c_direct = poly_mult_direct(a, b, p)
    end = time.time()

    print(f"Direct of degree {deg} took {end - start:.6f} seconds.")

    root = find_primitive_root(n, p)
    twiddles = gen_twiddles(n, root, p)
    inv_root = mod_inv(root, p)
    inv_twiddles = gen_twiddles(n, inv_root, p)

    # Benchmark NTT naive multiplication
    start = time.time()
    c_ntt_naive = poly_mult_ntt(a, b, p, root, twiddles, inv_root, inv_twiddles, ntt_naive, intt_naive)
    end = time.time()

    # print("real ntt of a:", sympy.ntt(a, p))
    # print("real ntt of b:", sympy.ntt(b, p))
    # sympy_c = [(x * y) % p for x, y in zip(sympy.ntt(a, p), sympy.ntt(b, p))]
    # print("real ntt of product:", sympy_c)
    # print("real intt of product:", sympy.intt(sympy_c, p))

    print(f"NTT Naive of degree {deg} took {end - start:.6f} seconds.")
 
    # Verify correctness
    if c_ntt_naive != c_direct:
        print("Mismatch between NTT naive multiplication and direct multiplication!")
        assert(False)
    else:
        print("NTT naive multiplication matches direct multiplication.")

    # Benchmark NTT butterfly multiplication
    start = time.time()
    c_ntt_butterfly = poly_mult_ntt(a, b, p, root, twiddles, inv_root, inv_twiddles, ntt_butterfly, intt_butterfly)
    end = time.time()
    
    print(f"NTT Butterfly of degree {deg} took {end - start:.6f} seconds.")
    
    # Verify correctness
    if c_ntt_butterfly != c_direct:
        print("Mismatch between NTT butterfly multiplication and direct multiplication!")
        assert(False)
    else:
        print("NTT butterfly multiplication matches direct multiplication.")

if __name__ == "__main__":
    bench_poly_mult()