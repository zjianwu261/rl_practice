from numpy.random import choice
samples = choice(['R', 'G', 'B'], 100, p=[0.2, 0.5, 0.3])
print(samples)