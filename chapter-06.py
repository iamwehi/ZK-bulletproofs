import random
from functools import reduce

import numpy as np
from py_ecc.bn128 import FQ, G1, Z1, add, eq, multiply
from py_ecc.bn128 import curve_order as p


def random_element():
    return random.randint(0, p)


def add_points(*points):
    return reduce(add, points, Z1)


# if points = G1, G2, G3, G4 and scalars = a,b,c,d vector_commit returns
# aG1 + bG2 + cG3 + dG4
def vector_commit(points, scalars):
    return reduce(add, [multiply(P, i) for P, i in zip(points, scalars)], Z1)


# these EC points have unknown discrete logs:
G_vec = [
    (
        FQ(
            6286155310766333871795042970372566906087502116590250812133967451320632869759
        ),
        FQ(
            2167390362195738854837661032213065766665495464946848931705307210578191331138
        ),
    ),
    (
        FQ(
            6981010364086016896956769942642952706715308592529989685498391604818592148727
        ),
        FQ(
            8391728260743032188974275148610213338920590040698592463908691408719331517047
        ),
    ),
    (
        FQ(
            15884001095869889564203381122824453959747209506336645297496580404216889561240
        ),
        FQ(
            14397810633193722880623034635043699457129665948506123809325193598213289127838
        ),
    ),
    (
        FQ(
            6756792584920245352684519836070422133746350830019496743562729072905353421352
        ),
        FQ(
            3439606165356845334365677247963536173939840949797525638557303009070611741415
        ),
    ),
]


# return a folded vector of length n/2 for scalars
def fold(scalar_vec, u):
    a = []
    for i in range(0, len(scalar_vec), 2):
        a.append(scalar_vec[i] * u + scalar_vec[i + 1] * pow(u, -1, p))

    return a


# return a folded vector of length n/2 for points
def fold_points(point_vec, u):
    a = []
    for i in range(0, len(point_vec), 2):
        a.append(
            add(multiply(point_vec[i], u), multiply(point_vec[i + 1], pow(u, -1, p)))
        )

    return a


# return L, R as a tuple
def compute_secondary_diagonal(G_vec, a):
    L = Z1
    R = Z1
    for i in range(len(a)):
        if i % 2 == 0:
            L = add(L, multiply(G_vec[i + 1], a[i]))
        else:
            R = add(R, multiply(G_vec[i - 1], a[i]))
    return (L, R)


a = [9, 45, 23, 42]

# prover commits
A = vector_commit(G_vec, a)
L, R = compute_secondary_diagonal(G_vec, a)

# verifier computes randomness
u = random_element()

# prover computes fold(a)
aprime = fold(a, u)

# verifier computes fold(G)
Gprime = fold_points(G_vec, pow(u, -1, p))

# verification check
assert eq(
    vector_commit(Gprime, aprime),
    add_points(multiply(L, pow(u, 2, p)), A, multiply(R, pow(u, -2, p))),
), "invalid proof"
assert len(Gprime) == len(a) // 2 and len(aprime) == len(a) // 2, (
    "proof must be size n/2"
)
