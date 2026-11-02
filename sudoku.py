###
# min_{x_{v,r,c}} 1
# s.t. x_{v,r,c} \in {0, 1}
# v \in 1..9
# r = 1..9
# c = 1..9


import cvxpy as cp


def main():
    objective = cp.Minimize(1) # min 1
    x = cp.Variable((9, 9, 9), boolean=True) # s.t. x_{v, r, c} \in {0, 1}
    constraints = []

    for a in range(9):
        for b in range(9):
            # a cell can have only one number
            constraints.append(cp.sum(x[a, b, :]) == 1)
            # each row has 1-9
            constraints.append(cp.sum(x[a, :, b]) == 1)
            # each column has 1-9
            constraints.append(cp.sum(x[:, a, b]) == 1)

    # each 3x3 has 1-9
    for v in range(9):
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                constraints.append(
                    cp.sum(x[br:br + 3, bc:bc + 3, v]) == 1
                )


    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    print(result)

    if x.value is not None:
        grid = x.value.argmax(axis=2) + 1
        print(grid)
        ### e.g.
        # [[2 5 4 9 8 6 7 1 3]
        # [7 9 1 4 5 3 2 6 8]
        # [6 8 3 7 2 1 5 9 4]
        # [5 3 7 8 1 4 9 2 6]
        # [9 1 8 2 6 7 3 4 5]
        # [4 2 6 3 9 5 8 7 1]
        # [1 7 2 5 4 8 6 3 9]
        # [3 6 5 1 7 9 4 8 2]
        # [8 4 9 6 3 2 1 5 7]]

if __name__ == "__main__":
    main()
