import argparse
import logging
import os
from timeit import default_timer as timer

import matplotlib
matplotlib.use('Agg') # Fixes weird segfaults, see http://matplotlib.org/faq/howto_faq.html#matplotlib-in-a-web-application-server

from fsm_effective_stress import compute_damage, compute_effective_stress
from fsm_load_modal_composites import load_modal_composites
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.lib.recfunctions import append_fields


__version__ = '1.0.1'


DEFAULT_CMAP = 'inferno'
FIGURE_SIZE = (11.7, 8.3) # In inches

SUBPLOTS_SPEC = [
    # key,        description
    ('D',         'damage'),
    ('D',         'damage (3D overview)'),
    ('sigma_eff', 'effective buckling stress [MPa]'),
    ('sigma_cr',  'elastic buckling stress [MPa]'),
]


def plot_modal_composite(modal_composites, column_units, column_descriptions):
    logging.info("Plotting modal composites...")
    start = timer()

    def _get_column_title(column_name):
        description = column_descriptions[column_name]
        unit = column_units[column_name]
        return description if not unit else "%s [%s]" % (description, unit)

    x = modal_composites['a']
    y = modal_composites['t_b']

    shape = np.unique(x).shape[0], np.unique(y).shape[0]
    X = x.reshape(shape)
    Y = y.reshape(shape)

    for ax_idx, (key, description) in enumerate(SUBPLOTS_SPEC, start=1):
        z = modal_composites[key]
        Z = z.reshape(shape)

        if '3D' in description:
            ax = plt.subplot(2, 2, ax_idx, projection='3d', elev=30.)
            ax.plot_wireframe(X, Y, Z, rcount=0, ccount=10)
        else:
            plt.subplot(2, 2, ax_idx)
            plt.imshow(
                Z.T,
                aspect='auto',
                interpolation='none',
                origin='lower',
                extent=[x.min(), x.max(), y.min(), y.max()],
            )
            plt.colorbar()

        plt.title(description)
        plt.xlabel(_get_column_title('a'))
        plt.ylabel(_get_column_title('t_b'))

    logging.info("Plotting completed in %f second(s)", timer() - start)

def configure_matplotlib(cmap=DEFAULT_CMAP):
    matplotlib.rc('figure',
        figsize=FIGURE_SIZE,
        titlesize='xx-large'
    )

    matplotlib.rc('figure.subplot',
        left   = 0.05, # the left side of the subplots of the figure
        right  = 0.99, # the right side of the subplots of the figure
        bottom = 0.06, # the bottom of the subplots of the figure
        top    = 0.95, # the top of the subplots of the figure
        wspace = 0.13, # the amount of width reserved for blank space between subplots
        hspace = 0.25, # the amount of height reserved for white space between subplots
    )

    matplotlib.rc('image',
        cmap=cmap
    )

def analyze_models(viscoelastic_model_file, elastic_model_file, report_file, **filters):
    with PdfPages(report_file) as pdf:
        elastic, column_units, column_descriptions = load_modal_composites(elastic_model_file, **filters)
        viscoelastic, _, _ = load_modal_composites(viscoelastic_model_file, **filters)

        omega = elastic['omega']
        omega_d = viscoelastic['omega']
        sigma_d = viscoelastic['sigma_cr']
        D = compute_damage(omega, omega_d)
        sigma_eff = compute_effective_stress(omega, omega_d, sigma_d)

        modal_composites = append_fields(elastic, names=['D', 'sigma_eff'], data=[D, sigma_eff], usemask=False)
        plot_modal_composite(modal_composites, column_units, column_descriptions)

        pdf.savefig()
        plt.close() # Prevents memory leaks

def main():
    # Setup command line option parser
    parser = argparse.ArgumentParser(
        description='Damage analysis and visualization of the parametric model '\
                    'of buckling and free vibration in prismatic shell '\
                    'structures, as computed by the fsm_eigenvalue project.',
    )
    parser.add_argument(
        'viscoelastic_model_file',
        help="File storing the computed viscoelastic parametric model"
    )
    parser.add_argument(
        '-e',
        '--elastic_model_file',
        metavar='FILENAME',
        help="File storing the computed elastic parametric model, determined from '<viscoelastic_model_file>' by default"
    )
    parser.add_argument(
        '-r',
        '--report_file',
        metavar='FILENAME',
        help="Store the analysis report to the selected FILENAME, uses '<viscoelastic_model_file>.pdf' by default"
    )
    parser.add_argument(
        '--a-min',
        metavar='VAL',
        type=float,
        help='If specified, clip the minimum strip length [mm] to VAL'
    )
    parser.add_argument(
        '--a-max',
        metavar='VAL',
        type=float,
        help='If specified, clip the maximum strip length [mm] to VAL'
    )
    parser.add_argument(
        '--t_b-min',
        metavar='VAL',
        type=float,
        help='If specified, clip the minimum base strip thickness [mm] to VAL'
    )
    parser.add_argument(
        '--t_b-max',
        metavar='VAL',
        type=float,
        help='If specified, clip the maximum base strip thickness [mm] to VAL'
    )
    parser.add_argument(
        '-c',
        '--cmap',
        metavar='CMAP',
        default=DEFAULT_CMAP,
        help="Plot figures using the selected Matplotlib CMAP, '%s' by default" % DEFAULT_CMAP
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_const',
        const=logging.WARN,
        dest='verbosity',
        help='Be quiet, show only warnings and errors'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_const',
        const=logging.DEBUG,
        dest='verbosity',
        help='Be very verbose, show debug information'
    )
    parser.add_argument(
        '--version',
        action='version',
        version="%(prog)s " + __version__
    )
    args = parser.parse_args()

    # Configure logging
    log_level = args.verbosity or logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    configure_matplotlib(cmap=args.cmap)

    if not args.elastic_model_file:
        args.elastic_model_file = args.viscoelastic_model_file.replace('viscoelastic', 'elastic')

    if not args.report_file:
        args.report_file = os.path.splitext(args.viscoelastic_model_file)[0] + '.pdf'

    analyze_models(
        viscoelastic_model_file=args.viscoelastic_model_file,
        elastic_model_file=args.elastic_model_file,
        report_file=args.report_file,
        a_min=args.a_min,
        a_max=args.a_max,
        t_b_min=args.t_b_min,
        t_b_max=args.t_b_max,
    )

if __name__ == '__main__':
    main()
