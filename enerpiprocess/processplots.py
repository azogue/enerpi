# -*- coding: utf-8 -*-
from itertools import cycle
import matplotlib.dates as mpd
import matplotlib.pyplot as plt
import pytz
import seaborn as sns
from enerpiplot.enerplot import write_fig_to_svg, tableau20


TZ = pytz.timezone('Europe/Madrid')
FS = (16, 10)
D_GRID = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3), 7: (3, 3), 8: (3, 3), 9: (3, 3),
          10: (4, 3), 11: (4, 3), 12: (4, 3)}


def _make_axes(num_subplots, size=6):
    shape = D_GRID[num_subplots]
    fsize = (shape[1] * size, shape[0] * size)
    fig = plt.figure(figsize=fsize)
    return [fig.add_subplot(int('{}{}{}'.format(*shape, i + 1))) for i in range(num_subplots)]


def plot_intervalos(df_interv, df_out,
                    with_fill_events=True, with_raw_scatter=True, with_vlines=True, with_level=False,
                    with_label_intervals=True, y_labels=50, level_column='level_group',
                    ax=None, size=6,
                    colors_big_events=tableau20[::2], major_fmt='%H:%M:%S', xlim=None, img_name=None):
    """
    Plot intervalos y big_events por separado, con distintos colores

    :param df_interv:
    :param df_out:

    :param with_fill_events:
    :param with_raw_scatter:
    :param with_vlines:
    :param with_level:
    :param with_label_intervals:

    :param y_labels:
    :param level_column:
    :param ax:
    :param size:
    :param colors_big_events:
    :param major_fmt:
    :param xlim:
    :param img_name:

    :return: matplotlib axes
    """
    if type(df_interv) is list:
        axes = _make_axes(len(df_interv), size=size)
        for i, (df_interv_i, ax) in enumerate(zip(df_interv, axes)):
            plot_intervalos(df_interv_i, df_out,
                            with_fill_events, with_raw_scatter, with_vlines, with_level,
                            with_label_intervals, y_labels, level_column,
                            ax, size, colors_big_events, major_fmt,
                            xlim[i] if xlim is not None else None, None)
        if img_name is not None:
            fig = plt.gcf()
            fig.tight_layout()
            write_fig_to_svg(fig, img_name, preserve_ratio=True)
        return axes
    elif ax is None:
        ax = _make_axes(1, size=size)[0]

    # cm = cycle(mpc.get_cmap('viridis').colors[::4])
    cm = cycle(colors_big_events)
    color_small_ev = 'grey'
    n_intervalos = len(df_interv)
    ymax = (df_out['wiener'].max() // 100) * 100 + 200
    if n_intervalos > 200:
        with_vlines = False
    interv_big = 0
    for interv, row in df_interv.iterrows():
        ts_ini, ts_fin, is_big, n_i = row['ts_ini'], row['ts_fin'], row['big_event'], row['n']
        df_i = df_out.loc[ts_ini:ts_fin]
        if not df_i.empty:
            color = next(cm) if is_big else 'k'
            lw = 1.25 if is_big else .5
            alpha = .8 if is_big else .5
            legend = 'E{:3d}'.format(interv) if is_big else ''
            if with_vlines:
                ax.vlines(df_i[df_i.is_init].index, 0, ymax,  # df_i[df_i.is_init].step_median * 2,
                          lw=.25, alpha=.5, color=color_small_ev, label='')
            if is_big:
                interv_big += 1
                if with_fill_events:
                    if level_column in df_out:
                        ax.fill_between(df_i.index, df_i['wiener'], y2=df_i[level_column],
                                        lw=0, alpha=.2, color=color, label='')
                    else:
                        ax.fill_between(df_i.index, df_i['wiener'], y2=0, lw=0, alpha=.2, color=color, label='')
                if with_raw_scatter and ('power' in df_out):
                    ax.scatter(df_i.index, df_i['power'], s=10, lw=0, alpha=.4, c=color, label='')
                if with_label_intervals and (len(df_i) > 30):
                    ax.annotate('{}'.format(interv), (df_i.index[len(df_i) // 2], y_labels),
                                xycoords='data', ha='center', va='bottom', color=color, fontsize=8, rotation=90,
                                bbox={'boxstyle': "round4,pad=0.2", 'ec': color, 'lw': 1, 'fc': 'grey', 'alpha': 0.2})
            elif with_raw_scatter and ('power' in df_out):
                ax.scatter(df_i.index, df_i['power'], s=6, lw=0, alpha=.3, c=color_small_ev, label='')
            df_i['wiener'].plot(ax=ax, lw=lw, alpha=alpha, color=color, label=legend)
    if with_level and (level_column in df_out):
        df_out.loc[df_interv.ts_ini[0]:df_interv.ts_fin[-1]][level_column].plot(ax=ax, lw=1.25,
                                                                                color=tableau20[6], ls='--', alpha=.7)
    ax.set_ylim((0, ymax))
    ax.set_xlabel("{:%d/%b'%y}".format(df_interv.ts_ini[len(df_interv) // 2]), fontsize=9, labelpad=3)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.xaxis.tick_bottom()
    ax.xaxis.set_major_formatter(mpd.DateFormatter(major_fmt, tz=TZ))
    xl = ax.get_xlim()
    if xl[1] - xl[0] < .25:
        ax.xaxis.set_minor_locator(mpd.MinuteLocator(tz=TZ))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    ax.xaxis.set_tick_params(labelsize=9, pad=3)
    # if 0 < interv_big < 30:
    #    plt.legend(loc='best')
    if img_name is not None:
        fig = plt.gcf()
        fig.tight_layout()
        write_fig_to_svg(fig, img_name, preserve_ratio=True)
    return ax
