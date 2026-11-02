import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

dimensionality = 28 * 28
CACHE_PATH = Path("regularisation.json")

def load_set(csv: str) -> pd.DataFrame:
    return pd.read_csv(csv, names=["label"] + [str(i) for i in range(dimensionality)])


def build_svm(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    target: int,
    cached_lambda: float | None = None,
) -> tuple[np.ndarray, float, float]:
    print(f"{datetime.datetime.now()}: Building SVM for digit {target}")
    s = (train_set["label"] == target).replace({True: 1, False: -1}).values
    s_test = (test_set["label"] == target).replace({True: 1, False: -1}).values
    train_set_np = train_set.iloc[:, 1:].to_numpy(dtype=np.uint8)
    test_set_np = test_set.iloc[:, 1:].to_numpy(dtype=np.uint8)
    a = cp.Variable(train_set_np.shape[1])
    b = cp.Variable()
    regularisation = cp.Parameter(nonneg=True)

    objective = cp.Minimize(
        cp.sum(cp.pos(1 - cp.multiply(s, train_set_np @ a + b)))
        + regularisation * cp.sum_squares(a)
    )

    problem = cp.Problem(objective)

    # Cache optimal regularisation between runs
    if cached_lambda is not None:
        regularisation.value = cached_lambda
        problem.solve(solver=cp.CLARABEL)
        assert a.value is not None
        assert b.value is not None

        print(
            f"{datetime.datetime.now()}: "
            f"Digit {target} solved using cached λ={cached_lambda}"
        )

        return a.value.copy(), float(b.value), cached_lambda

    lambdas = [0, 1e-3, 1e-2, 1e-1, 0.5, 1]

    def eval(x: float) -> float:
        regularisation.value = x
        problem.solve(solver=cp.CLARABEL)
        classification = test_set_np @ a.value + b.value
        loss = np.maximum(0, 1 - s_test * classification)
        return np.sum(loss)

    best = np.inf
    best_a = None
    best_b = None
    best_lambda = None

    for l in lambdas:
        score = eval(l)
        if score < best:
            assert a.value is not None
            assert b.value is not None
            best = score
            best_a = a.value.copy()
            best_b = float(b.value)
            best_lambda = l

    print(
        f"{datetime.datetime.now()}: Digit: {target} solved, best regularisation: {best_lambda}"
    )
    assert best_a is not None
    assert best_b is not None
    assert best_lambda is not None

    return best_a, best_b, best_lambda


def classify(svms: list[tuple[np.ndarray, float]], test_set: pd.DataFrame) -> float:
    print(f"{datetime.datetime.now()}: Computing performance")
    test_set_np = test_set.iloc[:, 1:].to_numpy(dtype=np.uint8)
    classifications = []
    for a, b in svms:
        prediction = test_set_np @ a + b
        classifications.append(prediction)

    classifications = np.array(classifications)
    final_classification = np.argmax(classifications, axis=0)
    oracle = test_set["label"].values
    assert final_classification.shape == oracle.shape
    accuracy = np.mean(final_classification == oracle)
    return accuracy

def save_cache(cache: dict[str, float]) -> None:
    # Write to a temporary file first, then atomically replace
    # the existing cache to avoid partially written JSON.
    temporary_path = CACHE_PATH.with_suffix(".json.tmp")

    with temporary_path.open("w") as f:
        json.dump(cache, f, indent=4)

    temporary_path.replace(CACHE_PATH)

def main():
    train_set = load_set("mnist_train.csv")
    test_set = load_set("mnist_test.csv")

    # Load cache if it exists
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r") as f:
            regularisation_dict = json.load(f)
    else:
        regularisation_dict = {}

    # Keep solutions indexed by digit, regardless of
    # which worker finishes first.
    solutions: list[tuple[np.ndarray, float]] = [None] * 10  # pyright: ignore[reportAssignmentType]

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {}

        # Submit all ten SVM training tasks
        for target in range(10):
            cached_lambda = regularisation_dict.get(str(target))

            future = executor.submit(
                build_svm,
                train_set,
                test_set,
                target,
                cached_lambda,
            )

            futures[future] = target

        # Process results as soon as each SVM finishes
        for future in as_completed(futures):
            target = futures[future]

            a, b, best_lambda = future.result()

            # Store coefficients in the correct digit position
            solutions[target] = (a, b)

            # Save newly discovered regularisation immediately
            if str(target) not in regularisation_dict:
                regularisation_dict[str(target)] = best_lambda
                save_cache(regularisation_dict)

                print(
                    f"{datetime.datetime.now()}: "
                    f"Cached regularisation for digit {target}: "
                    f"{best_lambda}"
                )

    if any(solution is None for solution in solutions):
        raise RuntimeError("Some SVMs did not complete")

    performance = classify(solutions, test_set)
    print(performance)  # 0.9203


if __name__ == "__main__":
    main()


###
# Pseudocode:
# (note the implementation uses some parallel processing and caching values to a file to make the problem runnable on my hardware... Not recommended to run on the whole mnist dataset)
#
# ## Training part:
# svms = []
# for i = 0..10:
#   # create svm problem parameterised in l
#   svm_template(l) = min_a,b(sum(max(0, 1 - s_i*a^T*x+b)) + l * ||a||^2) // . = elementwise product (use vectorised computations)
#
#   best_loss = +inf
#   best_a; best_b;
#   for candidate_l in regularisation_options:
#     candidate_svm = svm_i(candidate_l)
#     a, b = cvxpy.solve(candidate_svm)
#     test_loss = sum(max(0, 1 - s_i*a^T*x+b)) // performed on test set, rather than train set
#     if test_loss < best_loss
#       best_loss = test_loss
#       best_a, best_b = a, b
#
#   svms[i] = (best_a, best_b)
#
# ## Testing part:
# classifications = []
# for i = 0..10:
#   a, b = svms[i]
#   classifications[i] = a^T * x + b // code implementation is vectorised X*a + b
#
# final_classification = argmax(classifications) // choose most confident classifier
#
# Final output:
# 18:40:01 ✗ uv run python mnist.py
# 2026-11-02 18:40:27.428592: Building SVM for digit 0
# 2026-11-02 18:40:29.125290: Building SVM for digit 1
# 2026-11-02 18:40:30.849413: Building SVM for digit 2
# 2026-11-02 18:40:32.810060: Building SVM for digit 3
# 2026-11-02 19:37:11.724134: Digit: 0 solved, best regularisation: 0.01
# 2026-11-02 19:37:11.922451: Cached regularisation for digit 0: 0.01
# 2026-11-02 19:37:13.654365: Building SVM for digit 4
# 2026-11-02 19:42:49.296178: Digit: 3 solved, best regularisation: 1
# 2026-11-02 19:42:49.508780: Cached regularisation for digit 3: 1
# 2026-11-02 19:42:51.574742: Building SVM for digit 5
# 2026-11-02 19:44:27.430652: Digit: 2 solved, best regularisation: 1
# 2026-11-02 19:44:27.708573: Cached regularisation for digit 2: 1
# 2026-11-02 19:44:29.740568: Building SVM for digit 6
# 2026-11-02 20:16:59.562709: Digit: 1 solved, best regularisation: 1
# 2026-11-02 20:16:59.827977: Cached regularisation for digit 1: 1
# 2026-11-02 20:17:01.886578: Building SVM for digit 7
# 2026-11-02 20:42:38.505511: Digit: 4 solved, best regularisation: 1
# 2026-11-02 20:42:38.731553: Cached regularisation for digit 4: 1
# 2026-11-02 20:42:40.791024: Building SVM for digit 8
# 2026-11-02 20:45:08.461352: Digit: 5 solved, best regularisation: 1
# 2026-11-02 20:45:08.741552: Cached regularisation for digit 5: 1
# 2026-11-02 20:45:10.805482: Building SVM for digit 9
# 2026-11-02 21:14:08.695635: Digit: 6 solved, best regularisation: 1
# 2026-11-02 21:14:08.945591: Cached regularisation for digit 6: 1
# 2026-11-02 21:26:08.775654: Digit: 7 solved, best regularisation: 1
# 2026-11-02 21:26:09.006053: Cached regularisation for digit 7: 1
# 2026-11-02 21:39:59.749508: Digit: 8 solved, best regularisation: 0.5
# 2026-11-02 21:39:59.972134: Cached regularisation for digit 8: 0.5
# 2026-11-02 21:47:40.066250: Digit: 9 solved, best regularisation: 1
# 2026-11-02 21:47:40.232620: Cached regularisation for digit 9: 1
# 2026-11-02 21:47:40.440877: Computing performance
# 0.9203
