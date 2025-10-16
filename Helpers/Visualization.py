import matplotlib.pyplot as plt
import matplotlib

class MultiLivePlot:
    def __init__(self, nrows, ncols, titles=None, xlabel='X', ylabel='Y', display_window=-1):
        """
        nrows, ncols : grid dimensions
        titles       : list of subplot titles (length nrows*ncols) or None
        xlabel, ylabel: axis labels for all subplots
        """
        
        matplotlib.use('TkAgg')   # or 'Qt5Agg' for Qt backend
        plt.ion()
        self.nrows = nrows
        self.ncols = ncols
        self.fig, self.axes = plt.subplots(nrows, ncols, squeeze=False, figsize=(10, 3))
        self.lines = []
        self.data = []
        self.display_window = display_window

        total = nrows * ncols
        titles = titles or [''] * total

        for idx, ax in enumerate(self.axes.flatten()):
            line, = ax.plot([], [], marker='o')
            ax.set_title(titles[idx])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            self.lines.append(line)
            # each subplot has its own x/y lists
            self.data.append({'x': [], 'y': []})

        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()

    def update(self, index, x, y):
        """
        index : integer in [0 .. nrows*ncols-1]
        x, y  : new data point for subplot `index`
        """
        d = self.data[index]
        d['x'].append(x)
        d['y'].append(y)

        line = self.lines[index]
        if self.display_window == -1:
            line.set_data(d['x'], d['y'])
        else:
            line.set_data(d['x'][-self.display_window:], d['y'][-self.display_window:])

        ax = self.axes.flatten()[index]
        ax.relim()
        ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
