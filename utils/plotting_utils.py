import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from color import ColorPalette
import numpy as np

class Plot:
    def __init__(self, data, z, alpha, patterns, window_size, save_dir=None):
        self.save_dir = save_dir
        self.color_palette = ColorPalette.get_palette()
        self.data = data
        self.z = z
        self.alpha = alpha
        self.window_size = window_size
        self.n_patterns = self.z[0].shape[1]
        self.colors = list()
        self.patterns = patterns

        for i in range(self.n_patterns):
            self.colors.append(self.color_palette[i]) 

    def plot_patterns_analysis(self, figsize=(16, 8), save_dir=None):

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self.data, c='black')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # # ax.spines['bottom'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        # # ax.get_xaxis().set_ticks([])
        # ax.get_yaxis().set_ticks([])
        i = 0
        for zs in self.z:
            zs = np.flip(zs,axis=1)
            _, z = np.where(zs == np.max(zs))
            alpha = np.max(zs)
            ax.axvspan(i, i+self.window_size-1, color=self.colors[z[0]], alpha=alpha)
            i += self.window_size
        # Set y-axis label
        ax.set_ylabel('Data')

        # Set title
        ax.set_title('Time Series Patterns with T2P Analysis', fontsize=18)
    
        if save_dir:
            self._save_fig(fig, save_dir, 'patterns_analysis')
        else:
            plt.show()

    def _plot_bar(self, name, y, figsize):
        # Set parameters
        x = [str(i) for i in np.arange(0,self.n_patterns)] # X-axis values


        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        if name == 'z':
            ax.bar(x, y, align='center', alpha=0.8, color=self.color_palette[-1], edgecolor='white', linewidth=2)
        elif name == 'alpha':
            ax.bar(x, y, align='center', alpha=0.8, color=self.color_palette[0], edgecolor='white', linewidth=2)
        if name == 'z':
            ax.set_ylim(0, 1)
        elif name == 'alpha':
            ax.set_ylim(0, max(y))


        # Add text annotations
        for i in range(len(x)):
            ax.text(i, y[i]+0.05, float("{:.1f}".format(y[i])), ha='center', fontsize=18, fontweight='bold')

        # Add annotations and shading
        if name == 'z':
            ax.annotate('z', xy=(0.5, -0.15), xycoords='axes fraction', fontsize=18, ha='center', va='center')
        elif name == 'alpha':
            ax.annotate(r'$\alpha$', xy=(0.5, -0.15), xycoords='axes fraction', fontsize=18, ha='center', va='center')
    

        # Customize the plot
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.xaxis.set_tick_params(width=0)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

        return fig

    def plot_z(self, index, figsize=(16, 8), save_dir=None):
      
        fig = self._plot_bar(name='z', y=np.flip(self.z[index][0]), figsize=figsize)

        if save_dir:
            self._save_fig(fig, save_dir, 'z_'+str(index))
        else:
            plt.show()

    def plot_alpha(self, index, figsize=(16, 8), save_dir=None):
        fig = self._plot_bar(name='alpha', y=np.flip(self.alpha[index][0]), figsize=figsize)

        if save_dir:
            self._save_fig(fig, save_dir, 'alpha_'+str(index))
        else:
            plt.show()

    def plot_patterns(self,save_dir):

        min_ = np.min(self.patterns)
        max_ = np.max(self.patterns) + abs(min_/2)
        min_ -= abs(min_/2)

        # Set the figure size
        fig = plt.figure(figsize=(10*self.n_patterns, 10))

        # Create a grid of n_patterns plots
        grid = plt.GridSpec(1,self.n_patterns, hspace=0.4)

        for i in range(self.n_patterns):
            # Create a subplot for the patterns
            ax = fig.add_subplot(grid[:, i])
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_ylim(min_,max_)
            ax.plot(self.patterns[0,i,:,0],linewidth=3,c='black')
            ax.set_facecolor(self.colors[i])
            ax.set_title(f"Pattern {i}")

        if save_dir:
            self._save_fig(fig, save_dir, 'patterns')
        else:
            plt.show()



    def _save_fig(self, fig, save_dir, filename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{filename}.png")
        fig.savefig(save_path,bbox_inches="tight",transparent=False,dpi=150)
        print(f"Plot saved to {save_path}")


def plot_line(losses, title="Training Loss", xlabel="Epoch", ylabel="Loss", save_dir=None):
    """
    Plot a line graph

    Args:
    - losses (list or numpy array): List or array of training losses.
    - title (str): Title of the plot (default: "Training Loss").
    - xlabel (str): Label for the x-axis (default: "Epoch").
    - ylabel (str): Label for the y-axis (default: "Loss").
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    formatter = ticker.FormatStrFormatter('%1.2f')
    ax.yaxis.set_major_formatter(formatter)
    ax.plot(losses, color="#000099", linewidth=1.5)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.set_tick_params(width=0.5)
    ax.yaxis.set_tick_params(width=0.5)
    
    plt.xticks(np.arange(0, len(losses), step=max(1, len(losses) // 10)))
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"{title}.png")
        fig.savefig(save_path,bbox_inches="tight",transparent=True,dpi=150)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
