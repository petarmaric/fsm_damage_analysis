About
=====

Console app and Python API for damage analysis and visualization of the
parametric model of buckling and free vibration in prismatic shell structures,
as computed by the `fsm_eigenvalue project`_.

This work is a part of the investigation within the research project
[ON174027]_, supported by the Ministry for Science and Technology, Republic of
Serbia. This support is gratefully acknowledged.

References
----------

.. [ON174027]
   "Computational Mechanics in Structural Engineering"

.. _`fsm_eigenvalue project`: http://bitbucket.org/petar/fsm_eigenvalue

Installation
============

To install fsm_damage_analysis run::

    $ pip install fsm_damage_analysis

Console app usage
=================

Quick start::

    $ fsm_damage_analysis <filename>

Show help::

    $ fsm_damage_analysis --help

Python API usage
================

Quick start::

    >>> import logging
    >>> logging.basicConfig(level=logging.DEBUG)

    >>> from fsm_damage_analysis import analyze_models, configure_matplotlib

    >>> viscoelastic_model_file = 'examples/barbero-viscoelastic.hdf5'
    >>> elastic_model_file = 'examples/barbero-elastic.hdf5'
    >>> report_file = viscoelastic_model_file.replace('.hdf5', '.pdf')

    >>> configure_matplotlib()
    >>> analyze_models(viscoelastic_model_file, elastic_model_file, report_file)

Contribute
==========

If you find any bugs, or wish to propose new features `please let us know`_.

If you'd like to contribute, simply fork `the repository`_, commit your changes
and send a pull request. Make sure you add yourself to `AUTHORS`_.

.. _`please let us know`: https://bitbucket.org/petar/fsm_damage_analysis/issues/new
.. _`the repository`: http://bitbucket.org/petar/fsm_damage_analysis
.. _`AUTHORS`: https://bitbucket.org/petar/fsm_damage_analysis/src/default/AUTHORS
