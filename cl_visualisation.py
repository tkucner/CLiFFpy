import matplotlib.pyplot as plt


def plot_data_simple(data):
    x = data[:, 0]
    y = data[:, 1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(x, y, s=5)
    ax.set_xlabel('th')
    ax.set_ylabel('rho')
    #plt.colorbar(scatter)
    plt.show()
    #fig.savefig("mean_shift_result")


