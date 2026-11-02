import cvxpy as cp
import numpy as np
import pandas as pd

dimensionality = 28 * 28


def load_set(csv: str) -> pd.DataFrame:
    return pd.read_csv(csv, names=["label"] + [str(i) for i in range(dimensionality)])


def build_svm(
    train_set: pd.DataFrame, test_set: pd.DataFrame, target: int
) -> tuple[np.ndarray, np.ndarray]:
    s = (train_set["label"] == target).replace({True: 1, False: -1}).values
    s_test = (train_set["label"] == target).replace({True: 1, False: -1}).values
    train_set_np = train_set.iloc[:, 1:].to_numpy(dtype=np.uint8)
    test_set_np = train_set.iloc[:, 1:].to_numpy(dtype=np.uint8)
    a = cp.Variable(train_set_np.shape[1])
    b = cp.Variable()
    regularisation = cp.Parameter(nonneg=True)

    objective = cp.Minimize(
        cp.sum(cp.pos(1 - cp.multiply(s, train_set_np @ a + b)))
        + regularisation * cp.sum_squares(a)
    )

    problem = cp.Problem(objective)

    lambdas = [0, 1e-3, 1e-2, 1e-1, 0.5, 1]

    def eval(x: float) -> float:
        regularisation.value = x
        problem.solve(solver=cp.CLARABEL)
        classification = test_set_np @ a.value + b.value
        loss = np.max([np.zeros_like(s_test), 1 - s_test * classification])
        return np.sum(loss)

    best = np.inf
    best_a = None
    best_b = None
    assert len(lambdas) != 0
    for l in lambdas:
        score = eval(l)
        if score < best:
            best = score
            best_a = a.value
            best_b = b.value

    return best_a, best_b  # pyright: ignore[reportReturnType] - because lambdas is not empty


def classify(svms: list[tuple[np.ndarray, np.ndarray]], test_set: pd.DataFrame):
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


def main():
    train_set = load_set("mnist_train.csv")
    test_set = load_set("mnist_test.csv")

    solutions = [build_svm(train_set.head(5000), test_set, i) for i in range(10)]

    performance = classify(solutions, test_set)
    print(performance)


if __name__ == "__main__":
    main()


###
# Pseudocode:
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
