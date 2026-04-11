from pathlib import Path
import pickle

# cesta k jednej úlohe
problem_path = Path("data_sorted/2015_problems/preprocessed").glob("*.p")

# vezmeme prvý súbor
problem_file = next(problem_path)

with open(problem_file, "rb") as f:
    problem = pickle.load(f)

print("Loaded file:", problem_file)
print("Identifier:", problem.identifier)
print("Initial X shape (.xx):", problem.xx.shape)
print("Initial y shape (.yy):", problem.yy.shape)
print("Domain shape:", problem.domain.shape)
print("Labels shape:", problem.labels.shape)
print("Maximum:", problem.maximum)
print("Maximiser:", problem.maximiser)