
import numpy as np
import pandas as pd

from matplotlib import pyplot
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


def agg_assign(gb, fdict):
    """Auxiliar function (proxy method to be passed to `pipe`)."""
    data = {
        (cl, nm): gb[cl].agg(fn)
        for cl, d in fdict.items()
        for nm, fn in d.items()
    }
    pd.options.display.float_format = '{:.0f}'.format
    return pd.DataFrame(data)


def univariate_analysis(df, feature, flag, bins=None, precision=2, detail=True):
    """ Univarate analysis

    The feature is binned in multiple intervals and for each bin is shown:
        - % of flag (mean of flag)
        - n (count of observations)
        - n1 (count of flags)

    Args:
        df          : Dataset (pandas.DataFrame).
        feature     : Name of the feature.
        flag        : Name of the flag.
        bins        : Either an integer (number of bins) or a list with predefined cuts.
        precision   : Decimal precision at generating the bins. Default is 2.
        detail      : If True, it prints a table with the detail for each bin.

    """
    if bins:
        s = pd.concat([pd.cut(df[feature], bins=bins, precision=precision), df['flag']], axis=1).groupby(feature)['flag'].mean().rename(columns={'flag': 0})
        s_n1 = pd.concat([pd.cut(df[feature], bins=bins, precision=precision), df['flag']], axis=1).groupby(feature)['flag'].sum().rename(columns={'flag': 1})
        s_n = pd.concat([pd.cut(df[feature], bins=bins, precision=precision), df['flag']], axis=1).groupby(feature)['flag'].count().rename(columns={'flag': 2})

        if detail:
            #print("\n%s" % feature)
            #print("-"*50)
            print(pd.concat([s, s_n1, s_n], axis=1).rename(columns={0:'% flag', 1:'n1', 2:'n'}))

        indices = np.argsort(s.values)
        bins = [(str(interval)) for interval in s.index.values]

        fig, ax1 = plt.subplots(figsize=(14,7), facecolor='w')

        # axis 1 (mean of flag)
        ax1.plot(range(len(indices)), s.values, '-o', ms=10, lw=2, alpha=1)
        ax1.set_title(feature+"\n")
        ax1.set_xticks(range(len(bins)))
        ax1.set_xticklabels(bins, fontsize=12)
        ax1.set_xlabel('bins')
        ax1.set_ylabel('% flag')
        ax1.tick_params('y')

        labels = ['{:.2%}'.format(v) for v in s.values]
        for i in range(len(indices)):
            if ~np.isnan(s.values[i]):
                plt.text(i, s.values[i] + 0.01,
                         str(labels[i]),
                         fontsize=12)

        # axis 2 (total count)
        ax2 = ax1.twinx()
        ax2.bar(range(len(indices)), s_n.values, align='center', alpha=0.2, color='g')
        ax2.set_ylabel('# obs', color='g')
        ax2.tick_params('y', colors='g')
        ax2.grid(False)

        rects = ax2.patches
        labels = ['{:,}'.format(v) for v in s_n.values]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width()/ 2, height + height*0.01,
                    label,
                    color='g',
                    alpha=0.5,
                    fontsize=12,
                    ha='center',
                    va='bottom')

        plt.show()
