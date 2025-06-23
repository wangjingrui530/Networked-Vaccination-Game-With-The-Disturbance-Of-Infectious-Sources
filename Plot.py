import matplotlib.pyplot as plt


def Visualization(x, y, xlabel, ylabel, title, label, path, multiple):
    if multiple:
        fig1 = plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        for index in range(len(y)):
            plt.plot(x, y[index], label=label[index])
        plt.legend()
        fig1.savefig(path)
        # plt.show()
    else:
        fig1 = plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x, y)
        fig1.savefig(path)
        # plt.show()


if __name__ == "__main__":
    print("plot")
    Visualization(list(range(100)), [[i*i for i in range(100)], [i*i*i for i in range(100)]],
                  'xlabel', 'ylabel', 'title', ['y0', 'y1'], 'multi-lines.jpg', multiple=True)
    Visualization(list(range(100)), [i*i for i in range(100)], 'xlabel',
                  'ylabel', 'title', None, 'single-line.jpg', False)