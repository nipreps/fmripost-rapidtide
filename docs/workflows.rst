.. include:: links.rst

===========================
Processing pipeline details
===========================

*fMRIPost-rapidtide* adapts its pipeline depending on what data and metadata are
available and are used as the input.
For example, slice timing correction will be
performed only if the ``SliceTiming`` metadata field is found for the input
dataset.

A (very) high-level view of the simplest pipeline is presented below:

.. workflow::
    :graph2use: orig
    :simple_form: yes

    from fmripost_rapidtide.workflows.tests import mock_config
    from fmripost_rapidtide.workflows.base import init_single_subject_wf

    with mock_config():
        wf = init_single_subject_wf('01')
