import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import cho_factor, cho_solve


class Solver:
    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> None:
        self.A = A
        self.b = b
        self.c = c
        self.m, self.n = A.shape
        n, m = self.n, self.m

        self.F = np.block(
            [
                [A, np.zeros((m, m)), np.zeros((m, n))],
                [c.T, b.T, np.zeros((1, n))],
                [np.zeros((n, n)), A.T, -1 * np.eye(n)]
            ]
        )
        self.g = np.vstack([
            b,
            0,
            -1 * c
        ])
        FF = self.F @ self.F.T
        self.FF_factor = cho_factor(FF)
        assert self.F.shape == (self.m + 1 + self.n, 2 * self.n + self.m)
        assert self.g.shape == (self.m + 1 + self.n, 1)


    def project_onto_c(self, z: np.ndarray) -> np.ndarray:
        x = z[0:self.n, :]
        nu = z[self.n:self.n+self.m, :]
        lam = z[self.n+self.m:, :]
        projected = np.vstack([
            np.max([np.zeros_like(x), x], axis=0),
            nu,
            np.max([np.zeros_like(lam), lam], axis=0),
        ])
        assert(projected.shape == z.shape)
        return projected

    def project_onto_A(self, z: np.ndarray) -> np.ndarray:
        y = 2 * self.F @ z - 2 * self.g
        nu = cho_solve(self.FF_factor, y) # re-using cholesky factorisation speeds up the code by about 6x on my machine compared to np.linalg.solve
        projected = z - self.F.T @ nu * 0.5
        assert(projected.shape == z.shape)
        return projected

    def iterate(self, z: np.ndarray) -> tuple[float, np.ndarray]:
        z1 = self.project_onto_A(z)
        z2 = self.project_onto_c(z1)
        residual = np.linalg.norm(z1 - z2, 2)
        return residual, z2

    def solve(self) -> tuple[float, np.ndarray]:
        z = np.zeros((self.n+self.m+self.n, 1))
        residual = np.inf
        iter = 0
        residuals = []
        while residual > 1e-6 and iter < 1000:
            residual, z = self.iterate(z)
            residuals.append(residual)
            iter += 1

        x = z[0:self.n, :]
        print(f"Complete after {iter} iterations, {residual=}. {x=}")
        plt.plot(residuals)
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.savefig("residuals.png")

        return residual, x


def gen_random_problem(n: int, m: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((m, n))
    c = rng.uniform(0, 1, size=(n, 1))

    # Ensure Ax = b has at least one nonnegative feasible solution.
    x_feas = rng.uniform(0.0, 1.0, size=(n, 1))
    b = A @ x_feas

    return A, b, c

def main():
    A, b, c = gen_random_problem(500, 100)
    solver = Solver(A, b, c)
    solver.solve()

if __name__ == "__main__":
    main()


###
# Pseudocode:
# F = [A   0   0
#      c^T b^T 0
#      0   A^T -I]
# g = [b 0 -c]^T
# FF_factor = cholesky_factor(F*F^T)
#
# z = 0
# while not converged:
#   # project onto A
#   nu = // solve F*F^T*nu = 2F*z - 2g using cholesky factorisation FF_factor
#   z = z - 0.5 * F^T * nu
#
#   # project onto C
#   z = [max(0, x)
#        nu
#        max(0, lambda)]
