import numpy as np

def generate_data():
    # Generate random signal
    n = 1000
    n_nnz = 10
    hat_x = np . zeros ( n )
    nnz_idx = np . random . randint (0 , n , n_nnz )
    hat_x [ nnz_idx ] = np . random . randn ( n_nnz )
    # Samples
    n_samples = 100
    A = np . random . randn ( n_samples , n )
    y = A @ hat_x
    return A, y

def huber(y: np.ndarray) -> float:
    absy = np.abs(y)
    elements = np.where(absy <= 1, y * y, 2 * absy - 1)
    return np.sum(elements)

def d_huber(y: np.ndarray) -> np.ndarray:
    return np.where(np.abs(y) <= 1, 2 * y, np.where(y < 0, -2, 2))

def gradient_step(A: np.ndarray, y: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    gradient = A.T @ d_huber(A @ x - y)
    return x - alpha * gradient

def projection_onto_l1_ball(x: np.ndarray, N: float) -> np.ndarray:
    if np.sum(np.abs(x)) <= N:
            return x

    # we need to remove some theta from each element of x to project onto the L1 norm ball
    # use a bisection technique to find theta approximately, then scale
    low = 0
    high = np.max(np.abs(x))

    for _ in range(50):
        theta = (low + high) / 2
        projected = np.maximum(np.abs(x) - theta, 0)

        if np.sum(projected) > N:
            low = theta
        else:
            high = theta

    result = np.sign(x) * np.maximum(np.abs(x) - high, 0)
    return result

def main():
    N = 10
    A, y = generate_data()
    x = np.zeros(A.shape[1])

    tau = 0
    diff_norm = None
    for _ in range(10_000):
        alpha = 1/(1 + tau) # use a decreasing step size instead of fixed, this is guaranteed to converge if the problem is unconstrained...
        init_x = x
        x = gradient_step(A, y, x, alpha)
        x = projection_onto_l1_ball(x, N)
        # convergence check
        diff_norm = np.linalg.norm(init_x - x, 2)
        if diff_norm / alpha < 1e-6:
            break
        tau += 1

    print(f"Completed after {tau} iterations. {diff_norm=}")
    print(f"loss={huber(A@x - y)}")
    print(f"gradient={np.linalg.norm(A.T @ d_huber(A @ x - y), 2)}")
    print(f"||x||={np.sum(np.abs(x))}")


if __name__ == "__main__":
    main()


###
# Pseudocode:
# x_0 = 0
# tau = 0
# do:
#   alpha = 1/(1 + tau)
#   x_int = x_tau - alpha * gradient
#   x_tau+1 = x_int * N / ||x_int||_1
#   tau++
# while not converged
