import matplotlib.pyplot as plt

data_names = ('ted_B', 's3rmt3m3', 'thermomech_dM', 'parabolic_fem')
c_data = [0.003957, 0.023791, 0.686783, 5.768847]
py_data = [0.004019, 0.025165, 0.704579, 5.988630]

fig, axs = plt.subplots(2, 2)
axs_arr = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

for i in range(4):
    pl = axs_arr[i]
    pl.bar('C', c_data[i], label="C", color='g', width=0.6)
    pl.bar('Python', py_data[i], label="Python", color='b', width=0.6)
    pl.set_title(data_names[i])
    pl.set_ylabel('time comsumed (s)')

lines_labels = pl.get_legend_handles_labels()
lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
fig.legend(lines, labels)
plt.show()
