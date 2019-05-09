"""
=====================================
Metric measures :mod:`metric`
=====================================
.. currentmodule:: metric

This package provides a number of metric measures that e.g. can be used for testing
and/or evaluation purposes on two binary masks (i.e. measuring their similarity) or
distance between histograms.

Binary metrics (:mod:metric.binary`)
===========================================
Metrics to compare binary objects and classification results.

Compare two binary objects
**************************
 
.. module:: metric.binary

.. autosummary::
    :toctree: generated/
    
    dc
    jc
    hd
    asd
    assd
    precision
    recall
    sensitivity
    specificity
    true_positive_rate
    true_negative_rate
    positive_predictive_value
    ravd
    
Compare two sets of binary objects
**********************************

.. autosummary::
    :toctree: generated/
    
    obj_tpr
    obj_fpr
    obj_asd
    obj_assd
    
Compare to sequences of binary objects
**************************************

.. autosummary::
    :toctree: generated/
    
    volume_correlation
    volume_change_correlation
    
Image metrics (:mod:`metric.image`)
=========================================
Some more image metrics (e.g. `~medpy.filter.image.sls` and `~medpy.filter.image.ssd`)
can be found in :mod:`medpy.filter.image`. 

.. module:: medpy.metric.image
.. autosummary::
    :toctree: generated/
    
    mutual_information
    
Histogram metrics (:mod:`medpy.metric.histogram`)
=================================================

.. module:: medpy.metric.histogram
.. autosummary::
    :toctree: generated/
    
    chebyshev
    chebyshev_neg
    chi_square
    correlate
    correlate_1
    cosine
    cosine_1
    cosine_2
    cosine_alt
    euclidean
    fidelity_based
    histogram_intersection
    histogram_intersection_1
    jensen_shannon
    kullback_leibler
    manhattan
    minowski
    noelle_1
    noelle_2
    noelle_3
    noelle_4
    noelle_5
    quadratic_forms
    relative_bin_deviation
    relative_deviation
"""