from scipy import io
import matplotlib.pyplot as plt


PROBLEMS = ['ted_B', 's3rmt3m3', 'thermomech_dM', 'parabolic_fem']

fig, axs = plt.subplots(2, 2)
axs_arr = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]


for idx, problem in enumerate(PROBLEMS):
    path = f"./test_data/{problem}.mtx"
    matrix = io.mmread(path)
    pl = axs_arr[idx]
    pl.spy(matrix, markersize=1)
    pl.set_title(problem)


plt.show()
