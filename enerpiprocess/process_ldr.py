# -*- coding: utf-8 -*-
import datetime as dt
from io import BytesIO
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from scipy.signal import medfilt
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from enerpi.api import enerpi_data_catalog
from enerpi.base import timeit
from enerpiplot.enerplot import tableau20, write_fig_to_svg
from enerpiprocess.sunposition import sun_position
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from prettyprinting import *


LAT = 38.631463
LONG = -0.866402
ELEV_M = 500
TZ = pytz.timezone('Europe/Madrid')

FS = (16, 10)
# COLORMAP = plt.get_cmap('Paired')
COLORMAP = plt.get_cmap('spectral')
# COLORMAP = plt.get_cmap('viridis')


def _info_tiempos(cad):
    global tic
    toc = time()
    print_ok('--> {:35}\t OK\t [TOOK {:.3f} seconds]'.format(cad, toc - tic))
    tic = toc


def show_info_clustering(X, labels, n_clusters_, labels_true=None):
    print_magenta('Estimated number of clusters: %d' % n_clusters_)
    if labels_true is not None:
        print_ok("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print_ok("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print_ok("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print_ok("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
        print_ok("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))
    print_info("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
    print_cyan(pd.DataFrame(pd.Series(labels).value_counts().rename('Label members')).T)


def plot_clustering(X, labels, title=None,
                     core_samples_mask=None, cluster_centers_indices=None, cluster_centers=None, force_scatter=False,
                     ax=None):
    """Plot clustering results"""

    unique_labels = list(sorted(set(labels)))
    n_clusters = len(unique_labels)
    colors = COLORMAP(np.linspace(0, 1, n_clusters))
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=FS)
    if cluster_centers is None:
        cluster_centers = [None] * n_clusters
    for k, col, center in zip(unique_labels, colors, cluster_centers):
        class_member_mask = labels == k
        size_scatter = 5
        if k == -1:  # Black used for noise.
            col, size_scatter = 'k', 2
        if core_samples_mask is not None:
            xy = X[class_member_mask & core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', color=col, lw=0, markersize=10, alpha=.7)
            xy = X[class_member_mask & ~core_samples_mask]
            ax.plot(xy[:, 0], xy[:, 1], 'o', color=col, lw=0, markersize=size_scatter, alpha=.6)
        elif cluster_centers_indices is not None:
            cluster_center = X[cluster_centers_indices[k]]
            ax.plot(X[class_member_mask, 0], X[class_member_mask, 1], 'o', color=col, lw=0, markersize=size_scatter, alpha=.7)
            ax.plot(cluster_center[0], cluster_center[1], 'o', color=col, lw=0, markersize=10, alpha=.8)
            for x in X[class_member_mask]:
                ax.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], color=col, lw=1, alpha=.6)
        elif force_scatter or center is not None:
            ax.plot(X[class_member_mask, 0], X[class_member_mask, 1], 'o', color=col, lw=0, markersize=size_scatter, alpha=.7)
            if center is not None:
                ax.plot(center[0], center[1], 'o', color=col, lw=.5, markersize=10, alpha=.7)
        else:
            ax.plot(X[class_member_mask].T, c=col, alpha=.05)
    ax.axis('tight')
    if title is None:
        title = 'Number of clusters, k = {}'.format(n_clusters)
    ax.set_title(title, size=10)
    return ax


def _voronoi_image_mesh(model, X_reduced, step_mesh_size=.01):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    # step_mesh_size = .01  # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X_reduced[:, 0].min(), X_reduced[:, 0].max()
    y_min, y_max = X_reduced[:, 1].min(), X_reduced[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_mesh_size), np.arange(y_min, y_max, step_mesh_size))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    limits = xx.min(), xx.max(), yy.min(), yy.max()
    # print(x_min, x_max, y_min, y_max, limits)
    return Z, limits


def plot_clustering_as_voronoi(_X, labels, reduced_data, model, ax=None):
    Z, limits = _voronoi_image_mesh(model, reduced_data, step_mesh_size=.01)
    if ax is None:
        f, ax = plt.subplots(figsize=FS)
    ax.imshow(Z, interpolation='nearest', extent=limits, cmap=COLORMAP, alpha=.3, aspect='auto', origin='lower')
    s_labels = list(sorted(set(labels)))
    colors = COLORMAP(np.linspace(0, 1, len(s_labels)))
    for l, c in zip(s_labels, colors):
        idx = labels == l
        ax.scatter(reduced_data[idx, 0], reduced_data[idx, 1], color='k', s=10, alpha=.7)
    try:
        # Plot the centroids
        centroids = model.cluster_centers_
        ax.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=150, linewidths=3,
                    color='darkgrey', zorder=10, alpha=.9)
    except Exception as e:
        print_err('{} [{}] --> model={}'.format(e, e.__class__, model))
    ax.set_title('PCA-reduced data, k={}'.format(len(s_labels)))
    ax.set_xlim(*limits[:2])
    ax.set_ylim(*limits[2:])
    ax.set_xticks(())
    ax.set_yticks(())
    return ax


@timeit('plot_silhouette_analysis')
def _plot_silhouette_analysis(labels, X, ax=None):
    n_clusters = len(set(labels))
    silhouette_avg = metrics.silhouette_score(X, labels)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = metrics.silhouette_samples(X, labels)

    colors = COLORMAP(np.linspace(0, 1, n_clusters))
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=FS)
    ax.set_xlim([-0.1, 1])
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    y_lower = 10
    unique_labels = list(sorted(set(labels)))
    for k, color in zip(unique_labels, colors):
        if k == -1:
            color = 'k'
        ith_cluster_silhouette_values = sample_silhouette_values[labels == k]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(k))
        y_lower = y_upper + 10  # 10 for the 0 samples
    ax.set_title("Silhouette plot")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_title("Silhouette analysis (k = {})".format(n_clusters))
    return ax


def write_fig_to_png(fig, name_img):
    # plt.close(fig)
    canvas = FigureCanvas(fig)
    output = BytesIO()
    imgformat = 'png'
    canvas.print_figure(output, format=imgformat, transparent=True)
    svg_out = output.getvalue()
    try:
        with open(name_img, 'wb') as f:
            f.write(svg_out)
    except Exception as e:
        print_err('HA OCURRIDO UN ERROR GRABANDO SVG A DISCO: {}'.format(e))
        return False
    return True


@timeit('PLOT CLUSTERING', verbose=True)
def plot_clustering_figure(X_features, labels,
                           func_main_plot, *args_func_main_plot,
                           image_file=None, **kwargs_func_main_plot):

    feats_ax3 = np.c_[X_features[:, 0], X_features[:, 2:X_features.shape[1]], X_features[:, 1]]
    feats_ax4 = np.c_[X_features[:, 2:X_features.shape[1]], X_features[:, :2]]

    # PLOT ALL
    # plot propio de método + silhouette analysis + plot clusters v1/v2 + plot clusters v3/v4
    fig = plt.figure(figsize=(18, 18))
    shape_grid = (10, 10)

    ax1 = plt.subplot2grid(shape_grid, (0, 0), 6, 6)
    func_main_plot(X_features, labels, *args_func_main_plot, **kwargs_func_main_plot, ax=ax1)

    ax2 = plt.subplot2grid(shape_grid, (0, 6), 6, 4)
    _plot_silhouette_analysis(labels, X_features, ax=ax2)

    ax3 = plt.subplot2grid(shape_grid, (6, 0), 4, 5)
    plot_clustering(feats_ax3, labels, ax=ax3, force_scatter=True)

    ax4 = plt.subplot2grid(shape_grid, (6, 5), 4, 5)
    plot_clustering(feats_ax4, labels, ax=ax4, force_scatter=True)

    fig.tight_layout()
    if image_file is not None:
        if image_file.lower().endswith('png'):
            write_fig_to_png(fig, image_file)
        else:
            write_fig_to_svg(fig, image_file)
        plt.close()
    else:
        plt.show()


def plot_events_on_day(eventos, data_ldr_mf_step):
    delta = pd.Timedelta('2min')
    delta_big = pd.Timedelta('10min')
    ax, day_plot = None, None
    for t, row in eventos.iterrows():
        print_secc(str(t))
        roundings = data_ldr_mf_step.loc[t-delta:t+delta].copy()
        roundings_big = data_ldr_mf_step.loc[t-delta_big:t+delta_big].copy()
        if not roundings.empty:
            if ax is None:
                day_plot = t.date()
                print_red(day_plot)
            else:
                t = dt.datetime.combine(day_plot, t.time())
                new_index = pd.DatetimeIndex([dt.datetime.combine(day_plot, t) for t in roundings.index.time])
                new_index_big = pd.DatetimeIndex([dt.datetime.combine(day_plot, t) for t in roundings_big.index.time])
                roundings.index = new_index
                roundings_big.index = new_index_big
            if ax is None:
                ax = roundings['ldr'].plot(figsize=FS, lw=1, alpha=.8, color=tableau20[1])
            else:
                roundings['ldr'].plot(ax=ax, lw=1, alpha=.8, color=tableau20[1])
            ini = roundings['median_filter'].loc[:t]
            fin = roundings['median_filter'].loc[t:]
            if not ini.empty:
                ini.plot(ax=ax, lw=1.5, alpha=.8, color=tableau20[6])
            else:
                print_magenta('No hay INIT')
            if not fin.empty:
                fin.plot(ax=ax, lw=1.5, alpha=.8, color=tableau20[4])
            else:
                print_red('No hay FIN')
            ax.vlines([t], 0, 800, lw=1, linestyle='--', alpha=.6)
            roundings_big.ldr.plot(ax=ax, lw=.5, alpha=.5, color=tableau20[1])
    plt.show()


def plot_detalle_filtros(data_day, t0, tf):
    df = data_day.between_time(t0, tf)
    f, ax = plt.subplots(1, 1, figsize=FS)
    df.raw_ldr.plot(ax=ax, lw=1.5, alpha=.8, color=tableau20[0])
    df.median_filter.plot(ax=ax, lw=1.25, alpha=.8, color=tableau20[2])
    df.steps.plot(ax=ax, lw=1, alpha=.7, color=tableau20[8])
    df.steps_max.plot(ax=ax, lw=1, alpha=.8, color=tableau20[10], marker='o')
    plt.show()


def plot_barplot_typical_day_ldr(ldr_plot, delta_rs='10min', step_labels=6):
    ldr_10m = ldr_plot.resample(delta_rs).mean()
    ldr_10m['time'] = ldr_10m.index.time

    plt.figure(figsize=FS)
    ax = sns.barplot(x='time', y='ldr_median', data=ldr_10m.sort_values(by='time'), ci=90, estimator=np.median,
                     errcolor=tableau20[2], linewidth=0, facecolor=list(tableau20[3]) + [.3], capsize=1)
    for bar in ax.patches:
        x = bar.get_x()
        width = bar.get_width()
        centre = x + width / 2.
        bar.set_x(centre - 1 / 2.)
        bar.set_width(1.)
    for line in ax.lines:
        line.set_linewidth(.75)

    x_t = ax.get_xticks()
    x_tl = ax.get_xticklabels()
    ax.set_xticks(x_t[::step_labels] - .5)
    ax.set_xticklabels([t.get_text()[:-3] for t in x_tl[::step_labels]])

    cota_max = ldr_10m.groupby('time').ldr_min.min().reset_index(drop=True)
    cota_min = ldr_10m.groupby('time').ldr_max.max().reset_index(drop=True)
    ax.fill_between(cota_max.index, cota_min, cota_max, color=tableau20[6], facecolor=list(tableau20[7]) + [.2], linestyle='--', lw=2)
    #ldr_10m.groupby('time').ldr_min.min().reset_index(drop=True).plot(ax=ax, color=tableau20[2], ls='--', lw=2.5)
    #ldr_10m.groupby('time').ldr_max.max().reset_index(drop=True).plot(ax=ax, color=tableau20[4], ls='--', lw=2.5)
    return ax


# @timeit('kmeans_clustering', verbose=True)
def kmeans_clustering(X, k, km_init='k-means++', n_init=5, random_state=None, labels_true=None):
    # Kmeans:
    km = KMeans(init=km_init, n_clusters=k, n_init=n_init, random_state=random_state)
    km.fit(X)
    show_info_clustering(X, km.labels_, km.n_clusters, labels_true)
    return km.labels_, km.n_clusters, km


# @timeit('pca_clustering', verbose=True)
def pca_clustering(X, k, whiten=False, labels_true=None):
    """Principal component analysis
    """
    pca = PCA(n_components=2, whiten=whiten)
    reduced_data = pca.fit(X).transform(X)
    print_cyan('PCA Explained variance ratio: {} --> {}'
               .format(pca.explained_variance_ratio_, np.sum(pca.explained_variance_ratio_)))
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10) #, n_jobs=-1)
    kmeans.fit(reduced_data)
    show_info_clustering(reduced_data, kmeans.labels_, kmeans.n_clusters, labels_true)
    return kmeans.labels_, len(kmeans.cluster_centers_), kmeans, reduced_data


def dbscan_clustering(X_dbscan, eps=1.6, min_samples=5, labels_true=None):
    """DBSCAN Clustering
    """
    # Compute DBSCAN
    # X_dbscan = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_dbscan)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Show info:
    show_info_clustering(X_dbscan, labels, n_clusters_, labels_true)
    return db.labels_, n_clusters_, db, core_samples_mask


def affinity_propagation_clustering(X_std_scaler, preference=-250, labels_true=None):
    """Affinity Propagation Clustering
    """
    # X_std_scaler = StandardScaler().fit_transform(X)
    af = AffinityPropagation(preference=preference).fit(X_std_scaler)

    cluster_centers_indices = af.cluster_centers_indices_
    n_clusters_ = len(cluster_centers_indices)

    # Show info:
    show_info_clustering(X_std_scaler, af.labels_, n_clusters_, labels_true)

    return af.labels_, n_clusters_, af, cluster_centers_indices


def _extract_features_events(eventos_feat, data_ldr_mf_steps, features_around=10):
    # Feature as raw as waveform around event:
    # eventos_feat = eventos_tot[eventos_tot.es_subida]
    X_ag = np.zeros((len(eventos_feat), 4 * features_around + eventos_feat.shape[1]))
    for i, (t, row) in enumerate(eventos_feat.iterrows()):
        iloc = data_ldr_mf_steps.index.get_loc(t)
        xi_1 = data_ldr_mf_steps.iloc[iloc - features_around:iloc + features_around].values[:, 1]
        xi_2 = data_ldr_mf_steps.iloc[iloc - features_around:iloc + features_around].values[:, 2]
        xi_3 = row.sort_index().values
        X_ag[i, :] = np.concatenate([xi_3, xi_1, xi_2])
    return X_ag


def aglomerative_clustering(X, k=5, linkage='average', affinity='cityblock', labels_true=None):
    # Aglomerative Clustering:
    # linkage = "average", "ward", "complete"
    # metrics = ["euclidean", "cityblock", "cosine"]  #"manhattan",
    ag = AgglomerativeClustering(n_clusters=k, linkage=linkage, affinity=affinity).fit(X)

    show_info_clustering(X, ag.labels_, ag.n_clusters, labels_true)
    return ag.labels_, ag.n_clusters, ag


# TODO Pulir extracción de eventos y generar 'intervalos'
@timeit('extrae_eventos_LDR', verbose=True)
def extrae_eventos_LDR(data_raw, kernel_size=75, threshold_step=10, roll_number=7):
    _info_tiempos('Iniciando desde...')
    print_info("IDENTIFICACIÓN POI's LDR.\n * {} raw points. De {:%d-%m-%y} a {:%d-%m-%y} ({:.1f} horas)"
               .format(len(data_raw), data_raw.index[0], data_raw.index[-1], len(data_raw) / 3600.))

    # delta_rs = pd.Timedelta('5s')
    # DELTA_MIN_PARA_CONSIDERACION_MAX_LOCAL = 20

    # Resampling 1s
    data_homog = pd.DataFrame(data_raw.ldr.resample('1s').mean().fillna(method='bfill', limit=5).fillna(-1))
    _info_tiempos('Resampling 1s')

    # Median filter
    data_homog['median_filter'] = medfilt(data_homog.ldr, kernel_size=[kernel_size])
    _info_tiempos('MedFilt')

    # mf_resampled = set_sun_times(mf_resampled, delta_rs, tz=TZ, lat_deg=LAT, long_deg=LONG, offset='10min')

    # Saltos en eventos
    mf_roller = data_homog.median_filter.rolling(roll_number, center=True)
    ind_mid = roll_number // 2

    data_homog['steps'] = mf_roller.apply(
        lambda x: (x[roll_number - 1] + x[roll_number - 2]) / 2 - (x[1] + x[0]) / 2).fillna(0)
    _info_tiempos('Steps calc')
    # Busca saltos max-min
    roll_steps = data_homog['steps'].rolling(roll_number, center=True)
    idx_up = data_homog[
        roll_steps.apply(lambda x: (np.argmax(x) == ind_mid) and (x[ind_mid] > threshold_step)).fillna(0) > 0].index
    idx_down = data_homog[
        roll_steps.apply(lambda x: (np.argmin(x) == ind_mid) and (x[ind_mid] < -threshold_step)).fillna(0) > 0].index
    _info_tiempos('Steps UP / DOWN Filter')

    # cols_eventos = ['steps', 'median_filter', 'rms_err_min301']
    cols_eventos = ['steps', 'median_filter']
    subidas = data_homog.loc[idx_up, cols_eventos].copy()
    bajadas = data_homog.loc[idx_down, cols_eventos].copy()
    _info_tiempos('Eventos')

    mf_slope_roll = data_homog.median_filter.diff().rolling(roll_number, center=True).mean()
    mf_roll = mf_roller.median()

    offsets_mf = range(-11, 12, 3)
    for d in offsets_mf:
        subidas['slope_roll_{}'.format(d)] = mf_slope_roll.shift(d).loc[idx_up]
        bajadas['slope_roll_{}'.format(d)] = mf_slope_roll.shift(d).loc[idx_down]
        subidas['mf_roll_{}'.format(d)] = mf_roll.shift(d).loc[idx_up]
        bajadas['mf_roll_{}'.format(d)] = mf_roll.shift(d).loc[idx_down]

    assert (len(subidas.index.intersection(bajadas.index)) == 0)
    eventos = pd.concat([subidas, bajadas], axis=0).sort_index()
    eventos['es_subida'] = eventos['steps'] > 0
    _info_tiempos('Eventos slope & process')

    # Altitud / azimut solar
    eventos = eventos.join(sun_position(eventos.index, latitude_deg=LAT, longitude_deg=LONG, elevation=ELEV_M,
                                        delta_n_calc=1, south_orient=True).round(1))
    _info_tiempos('Eventos SUN position')
    # INTERVALOS
    # day.loc[eventos.index, 'interv'] = 1
    # day.interv = day.interv.fillna(0).cumsum()
    # day.groupby('interv').median_filter.count()
    print(eventos.shape)
    return eventos, data_homog


def _separa_eventos(eventos):
    eventos = eventos.T.sort_index().T.iloc[1:-1]
    eventos.es_subida = eventos.es_subida.astype(bool)
    evento_es_limite_exec = eventos.es_fin | eventos.es_init
    eventos_limite_init = eventos[evento_es_limite_exec & eventos.es_subida]
    eventos_limite_fin = eventos[evento_es_limite_exec & ~eventos.es_subida]
    subidas = eventos[~evento_es_limite_exec & eventos.es_subida]
    bajadas = eventos[~evento_es_limite_exec & ~eventos.es_subida]
    print_info("Separación de POI's:\n * TOTAL = {}; Inicios = {}; Finales = {};\n * SUBIDAS = {}; BAJADAS = {}; "
               .format(len(eventos), len(eventos_limite_init), len(eventos_limite_fin), len(subidas), len(bajadas)))
    return dict(subidas=subidas, bajadas=bajadas, limites_init=eventos_limite_init, limites_fin=eventos_limite_fin)


def compara_clustering_methods(lista_eventos, lista_n_clusters=(7, 7), names_eventos=('subidas', 'bajadas')):
    # **CLUSTERING**
    print_secc('CLUSTERING por grupos, distintos métodos:\n')
    cols_cluster = ['steps', 'azimuth', 'median_filter', 'altitude',
                    # 'mf_roll_-8', 'mf_roll_-2', 'mf_roll_1', 'mf_roll_4', 'mf_roll_7',
                    'slope_roll_-5', 'slope_roll_-2', 'slope_roll_1', 'slope_roll_4', 'slope_roll_7']
    # cols_view = ['altitude', 'azimuth', 'es_fin', 'es_init', 'es_subida', 'median_filter',
    #              'mf_roll_-11', 'mf_roll_10', 'steps']
    labeling = []
    for eventos, k, label_ev in zip(lista_eventos, lista_n_clusters, names_eventos):
        print_secc('CLUSTERING grupo: {}'.format(label_ev))

        # Features:
        X = eventos[cols_cluster].astype(float).values
        X_features = _extract_features_events(eventos[cols_cluster], data_tot, features_around=5)
        X_std_scaler = StandardScaler().fit_transform(X_features)

        # PCA:
        print_yellowb('\n*** PCA:')
        labels_pca, n_clusters_pca, model_pca, X_pca_2c = pca_clustering(X_features, k, whiten=True)
        _info_tiempos('PCA Clustering')

        print_red('* Features shape: {}; PCA-reduced shape: {}; StandardScaler shape: {}. Looking for {} clusters...'
                  .format(X_features.shape, X_pca_2c.shape, X_std_scaler.shape, k))
        # print_cyan('* Features sample:\n{};\n* PCA-reduced sample:\n{};\n* StandardScaler sample\n{};'
        #            .format(X_features[:2, :], X_pca_2c[:2, :], X_std_scaler[:2, :]))
        _info_tiempos('Features for {} samples'.format(len(eventos)))

        # KMEANS:
        print_yellowb('\n*** KMEANS++:')
        labels_km, n_clusters_km, model_km = kmeans_clustering(X, k)
        centers_km = model_km.cluster_centers_
        _info_tiempos('KMeans++ Clustering')

        # DBSCAN:
        print_yellowb('\n*** DBSCAN:')
        (labels_dbscan, n_clusters_dbscan,
         model_dbscan, core_samples_mask_dbscan) = dbscan_clustering(X_std_scaler, eps=1.8, min_samples=4,
                                                                     labels_true=None)
        _info_tiempos('DBSCAN Clustering')

        # Affinity propagation:
        print_yellowb('\n*** Affinity propagation:')
        (labels_afp, n_clusters_afp,
         model_afp, cluster_centers_indices_afp) = affinity_propagation_clustering(X_std_scaler, preference=-250,
                                                                                   labels_true=None)
        _info_tiempos('Affinity propagation Clustering')

        # Aglomerative Clustering:
        print_yellowb('\n*** Aglomerative Clustering:')
        # linkage = "average", "ward", "complete"
        # metrics = ["euclidean", "cityblock", "cosine"]  #"manhattan",
        # labels_ag, n_clusters_ag, model_ag = aglomerative_clustering(X_ag, k, linkage='average', affinity='cityblock')
        labels_ag, n_clusters_ag, model_ag = aglomerative_clustering(X_features, k, linkage='average',
                                                                     affinity='cityblock')
        _info_tiempos('Aglomerative Clustering')

        # PLOT ALL
        img_ext = 'svg'
        plot_clustering_figure(X_features, labels_km, plot_clustering, cluster_centers=centers_km,
                               image_file='eventos_{}_clustering_LDR_Kmeans.{}'.format(label_ev, img_ext))

        plot_clustering_figure(X_features, labels_pca, plot_clustering_as_voronoi, X_pca_2c, model_pca,
                               image_file='eventos_{}_clustering_LDR_PCA.{}'.format(label_ev, img_ext))

        plot_clustering_figure(X_std_scaler, labels_dbscan,
                               plot_clustering, core_samples_mask=core_samples_mask_dbscan,
                               image_file='eventos_{}_clustering_LDR_DBSCAN.{}'.format(label_ev, img_ext))

        plot_clustering_figure(X_std_scaler, labels_afp,
                               plot_clustering, cluster_centers_indices=cluster_centers_indices_afp,
                               image_file='eventos_{}_clustering_LDR_Affinity_prop.{}'.format(label_ev, img_ext))

        plot_clustering_figure(X_features, labels_ag,
                               plot_clustering, title="AgglomerativeClustering(affinity={})".format('cityblock'),
                               image_file='eventos_{}_clustering_LDR_Aglomerative.{}'.format(label_ev, img_ext))

        _info_tiempos('PLOT CLUSTERING')

        # ALL CLUSTERING LABELING
        labels = [labels_km, labels_pca, labels_dbscan, labels_afp, labels_ag]
        names = ['k_km', 'k_pca', 'k_dbscan', 'k_afp', 'k_ag']
        series = [pd.Series(labels, name=name, index=eventos.index) for labels, name in zip(labels, names)]
        df_clusters = pd.DataFrame(series).T
        df_clusters['super_cluster'] = 'cluster_{}_{}'.format(label_ev, k)
        _info_tiempos('df_clusters')
        print_ok(df_clusters.head())
        print_ok(df_clusters.tail())

        # Compara entre métodos:
        show_info_clustering(X_features, labels_km, len(set(labels_km)), labels_pca)
        show_info_clustering(X_features, labels_km, len(set(labels_km)), labels_dbscan)
        show_info_clustering(X_features, labels_km, len(set(labels_km)), labels_afp)
        show_info_clustering(X_features, labels_km, len(set(labels_km)), labels_ag)

        show_info_clustering(X_features, labels_pca, len(set(labels_pca)), labels_km)
        show_info_clustering(X_features, labels_pca, len(set(labels_pca)), labels_dbscan)
        show_info_clustering(X_features, labels_pca, len(set(labels_pca)), labels_afp)
        show_info_clustering(X_features, labels_pca, len(set(labels_pca)), labels_ag)

        labeling.append(df_clusters)
    return pd.DataFrame(pd.concat(labeling, axis=0)).sort_index()


if __name__ == '__main__':
    sns.set_style('whitegrid')
    path_st = 'eventos_LDR_clustering.h5'
    force_update = False
    tic = time()
    tic_ini = tic

    # Get POI's
    try:
        if force_update:
            raise FileNotFoundError
        eventos_tot = pd.read_hdf(path_st, 'eventos')
        data_tot = pd.read_hdf(path_st, 'data')
        _info_tiempos('READ Eventos')
    except (FileNotFoundError, KeyError):
        print_secc("*** Generación de POI's de LDR data:\n")

        # Catálogo y lectura de todos los datos.
        cat = enerpi_data_catalog()
        data, data_s = cat.get_all_data()
        LDR = pd.DataFrame(data.ldr).tz_localize(TZ)
        print_cyan(LDR.describe().T.astype(int))

        # Extracción de eventos
        eventos_tot, data_tot = extrae_eventos_LDR(LDR, threshold_step=10, roll_number=7)

        # sns.pairplot(eventos_tot, hue='es_subida')

        # Mod eventos:
        eventos_tot.loc[eventos_tot.altitude < -5, 'altitude'] = -5
        eventos_tot.loc[eventos_tot.altitude < -5, 'azimuth'] = -180
        eventos_tot['es_fin'] = False
        eventos_tot['es_init'] = False
        eventos_tot.loc[eventos_tot.mf_roll_10 == -1, 'es_fin'] = True
        eventos_tot.loc[eventos_tot['mf_roll_-11'] == -1, 'es_init'] = True
        _info_tiempos('Fin creación de eventos...')

        eventos_tot.to_hdf('eventos_LDR_clustering.h5', 'eventos')
        data_tot.to_hdf('eventos_LDR_clustering.h5', 'data')
        _info_tiempos('SAVED!...')

    # serie = data_tot.iloc[:10000].ldr
    # sm.graphics.tsa.plot_acf(serie, use_vlines=False, unbiased=False)
    #
    # arma = sm.tsa.ARMA(serie, (2, 2))
    # arma.score
    # arma.fit()
    # print_ok(arma)
    # print(arma.aic, arma.bic, arma.hqic)
    # print(arma.params)
    # print_magenta(sm.stats.durbin_watson(arma.resid.values))

    d_eventos = _separa_eventos(eventos_tot)
    grupos = ['subidas', 'bajadas', 'limites_init', 'limites_fin']
    n_ks = [7, 7, 2, 2]
    subidas = d_eventos['subidas']
    bajadas = d_eventos['bajadas']
    bajadas = d_eventos['limites_init']
    bajadas = d_eventos['limites_fin']
    _info_tiempos('Separación')

    cluster_multi_label = compara_clustering_methods([d_eventos[g] for g in grupos], n_ks, names_eventos=grupos)
    cluster_multi_label.to_hdf('eventos_multi_clustering.h5', 'clustering')
    eventos_tot.to_hdf('eventos_multi_clustering.h5', 'eventos')

    _info_tiempos('END')
    print_red('Total Execution TOOK {:.2f} secs'.format(time() - tic_ini))

    # subset_subidas = subidas[(subidas.km_pca == 6) & ~subidas.es_fin]
    # plot_events_on_day(subset_subidas, data_tot)
    # print_ok(subset_subidas.head(10))

    # ldr_minutos = data_tot.iloc[:,0].resample('1s').mean().fillna(method='ffill', limit=5)
    # ldr_minutos = ldr_minutos.fillna(method='ffill', limit=5).fillna(method='bfill', limit=5).fillna(-1)
    # ldr_minutos = ldr_minutos.resample('1min').median().round()
    # ldr_minutos = ldr_minutos[ldr_minutos > 0]
    #
    # x = ldr_minutos.index.hour * 60 + ldr_minutos.index.minute
    # y = ldr_minutos.values
    # ldr_max = ((ldr_minutos.max() + 50) // 50) * 50
    # _info_tiempos('Preparing for plots')

    # sns.set(style="ticks")
    # g = sns.jointplot(x, y, kind='kde', color="#4CB391", size=8, ratio=7, space=0, xlim=(0, 60*24), ylim=(0, ldr_max))
    # g.set_axis_labels('Horas', 'LDR level')
    # fig = plt.gcf()
    # write_fig_to_svg(fig, 'jointplot_ldr_kde.svg')
    # _info_tiempos('jointplot_ldr.svg SAVED')

    # from scipy.stats import kendalltau
    # g = sns.jointplot(x, y, kind='hex', color="#4CB391", size=8, ratio=7, space=0, stat_func=kendalltau,
    #                   xlim=(0, 60*24), ylim=(0, ldr_max), marginal_kws=dict(bins=144))
    # g.set_axis_labels('Horas', 'LDR level')
    # print(g)
    # fig = plt.gcf()
    # write_fig_to_svg(fig, 'jointplot_ldr_hex.svg')
    # _info_tiempos('jointplot_ldr.svg SAVED')
    #
    # fig = plt.gcf()
    # fig.clf()
    # ax = sns.barplot(x, y, color="#4CB391")
    # # g.set_axis_labels('Horas', 'LDR level')
    # write_fig_to_svg(fig, 'barplot_ldr.svg')
    # _info_tiempos('jointplot_ldr.svg SAVED')
    # # plt.show()




