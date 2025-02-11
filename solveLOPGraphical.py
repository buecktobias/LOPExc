from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog


@dataclass
class Constraint:
    coefficients: list[float]
    bound: float

    def __repr__(self) -> str:
        terms = " + ".join(f"{coef} * x{i + 1}" for i, coef in enumerate(self.coefficients))
        return f"Restriktion: {terms} ≤ {self.bound}"


@dataclass
class LOP:
    objective: list[float]
    constraints: list[Constraint]

    def __repr__(self) -> str:
        objective_str = " + ".join(f"{coef} * x{i + 1}" for i, coef in enumerate(self.objective))
        constraints_str = "\n".join(str(con) for con in self.constraints)
        return f"Maximiere: {objective_str}\nSubject to:\n{constraints_str}"


@dataclass
class Solution:
    optimal_point: tuple[float, float]
    objective_value: float

    def __repr__(self) -> str:
        return (f"Optimal Point: (x1, x2) = ({self.optimal_point[0]:.2f}, {self.optimal_point[1]:.2f})\n"
                f"Objective Value: {self.objective_value:.2f}")


def total_intersections(constraints: list[Constraint]) -> int:
    intersections = 0
    for i in range(len(constraints)):
        for j in range(i + 1, len(constraints)):
            if has_positive_intersection(constraints[i], constraints[j]):
                intersections += 1
    return intersections


def has_positive_intersection(con1: Constraint, con2: Constraint) -> bool:
    x, y = get_intersection_point(con1, con2)
    return x >= 0 and y >= 0


def get_intersection_point(con1: Constraint, con2: Constraint) -> tuple[float, float]:
    A = np.array([con1.coefficients, con2.coefficients])
    b = np.array([con1.bound, con2.bound])
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return -1, -1


def zero_safe(x: float) -> float:
    return x if x != 0 else 1e-20


def evaluate_constraint(constraint: Constraint, x_values: np.ndarray) -> np.ndarray:
    return (constraint.bound - constraint.coefficients[0] * x_values) / zero_safe(constraint.coefficients[1])


def solve_problem(problem: LOP) -> Solution:
    negative_objective = [-coefficient for coefficient in problem.objective]
    A = np.array([np.array(constraint.coefficients) for constraint in problem.constraints])
    upper_bound: np.array = np.array([constraint.bound for constraint in problem.constraints])
    # noinspection PyDeprecation
    result = linprog(negative_objective, A_ub=A, b_ub=upper_bound)

    if not result.success:
        raise ValueError("No solution found")

    return Solution(
            optimal_point=(result.x[0], result.x[1]),
            objective_value=-result.fun
    )



def plot_constraint(constraint: Constraint):
    x_values = np.linspace(0, 100, 10_000)
    y_values = evaluate_constraint(constraint, x_values)
    values = [(x, y) for x, y in zip(x_values, y_values) if x >= 0 and y >= 0]
    plt.plot(
            *zip(*values),
            label=f"{constraint.coefficients[0]}x1 + {constraint.coefficients[1]}x2 <= {constraint.bound}"
    )


def plot_solution(solution: Solution):
    optimal_point = solution.optimal_point
    plt.plot(optimal_point[0], optimal_point[1], 'ro', label="Optimal Solution")
    plt.annotate(
            f"({solution.optimal_point[0]:.2f}, {solution.optimal_point[1]:.2f})",
            solution.optimal_point, textcoords="offset points", xytext=(10, 10)
    )

def fill_feasible_region(constraints: list[Constraint]):
    x_values = np.linspace(0, 100, 10_000)
    y_values = np.minimum.reduce([evaluate_constraint(con, x_values) for con in constraints])
    y_values = np.maximum(y_values, 0)
    plt.fill_between(x_values, 0, y_values, alpha=0.2)


def plot_problem(problem: LOP, output_path: Path):
    plt.figure(figsize=(10, 8))

    for con in problem.constraints:
        plot_constraint(con)

    fill_feasible_region(problem.constraints)

    solution = solve_problem(problem)
    plot_solution(solution)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title('Linear Optimization Feasible Region and Optimal Solution')
    plt.grid(True)
    plt.savefig(output_path)

def save_problem_with_solution(problem: LOP, solution: Solution, output_path: Path):
    with open(output_path, 'w', encoding="utf-8") as file:
        file.write(str(problem) + "\n\n" + str(solution))