import random
from pathlib import Path

from openai import OpenAI

from solveLOPGraphical import Constraint, LOP, plot_problem, save_problem_with_solution, \
    solve_problem, \
    total_intersections

COEFFICIENT_NUMBERS = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BOUND_NUMBERS = [15, 20, 30, 50, 75, 100]


def random_constraint():
    coefficients = [random.choice(COEFFICIENT_NUMBERS), random.choice(COEFFICIENT_NUMBERS)]
    while sum(coefficients) == 0:
        coefficients = [random.choice(COEFFICIENT_NUMBERS), random.choice(COEFFICIENT_NUMBERS)]
    return Constraint(
            coefficients=coefficients,
            bound=random.choice(BOUND_NUMBERS)
    )


def get_objective_coefficient():
    return max(1, random.choice(COEFFICIENT_NUMBERS))


def random_objective():
    objective = [get_objective_coefficient(), get_objective_coefficient()]
    if sum(objective) == 0:
        return random_objective()
    return objective


def get_random_constraints(constraint_count: int, min_intersections: int):
    constraints = []
    for _ in range(constraint_count):
        constraint = random_constraint()
        constraints.append(constraint)

    while total_intersections(constraints) < min_intersections:
        return get_random_constraints(constraint_count, min_intersections)
    return constraints


def create_random_lop_problem():
    constraint_count = random.randint(2, 5)
    min_intersections = min(random.randint(1, 3), constraint_count - 1)
    constraints = get_random_constraints(constraint_count, min_intersections)
    return LOP(constraints=constraints, objective=random_objective())


def prettify(text: str):
    return text.replace(". ", ".\n")


def create_random_problem(result_graphic_path: Path, result_text_path: Path):
    random_lop = create_random_lop_problem()
    client = OpenAI()
    completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "developer",
                    "content": "Du bist ein Lehrer und erstellstst eine Lineare Optimierungsaufgabe."
                },
                {
                    "role": "developer", "content": f"""
                Schreibe den Text einer Linearen Optimierungsaufgabe. In der Aufgabe wird eine Planungsproblem beschrieben. 
                Es gibt mehrere Einschränkungen, die erfüllt werden müssen. Die Aufgabe besteht darin, die Zielfunktion zu maximieren.
                DIe Zielfunktion hat zwei Variablen. Die Einschränkungen sind lineare Ungleichungen.
                Überlege dir Namen für die passenden beiden Produkte die hergestellt werden sollen.
                Hier sind die Koeffizienten der Zielfunktion, sowie die Einschränkungen:
                {random_lop}
                
                Beschreibe das vorliegende Problem in einem Fließtext. Gebe keine Tabelle mit Hinweisen, gebe keine Lösungen oder Hinweise an.
                Gebe in dem Text nicht an, welche Werte Einschränkungen sind und auch nicht welche der Werte zur Zielfunktion gehören.
                Gebe ebenfalls keine Hinweise zu sinnvollen Variablen namen.
                Gebe die Zahlenwerte aus den Einschränkungen und der Zielfunktion an, sodass man die Aufgabe lösen kann,
                aber ohne exakt zu sagen, dass ein gewisser Wert eine Einschränkung oder die Zielfunktion ist.
                Gebe richtige Einheiten an, die zu den Zahlenwerten passen. Also z.B. Euro für Geld, Gewinn, Kosten, etc.
                Kg, gramm für Gewichte. Kubikmeter, Liter oder Milliliter für Volumen. Bei Anzahlen musst du nicht explizit Stück angeben.
                Flächen wie Quadratmeter, Quadratzentimeteer oder Hektar. Zeit in Stunden, Minuten oder Sekunden.
                """
                },

            ]
    )
    text = completion.choices[0].message.content

    solution = solve_problem(random_lop)
    plot_problem(random_lop, result_graphic_path)
    save_problem_with_solution(random_lop, solution, result_text_path)
    print(prettify(text))


if __name__ == '__main__':
    create_random_problem(
        Path("data") / "result_graphic.png",
        Path("data") / "result_text.txt"
        )
