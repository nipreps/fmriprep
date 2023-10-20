.. include:: links.rst

================
Developers - API
================
The *NiPreps* community and contributing guidelines
---------------------------------------------------
*fMRIPrep* is a *NiPreps* application, and abides by the
`NiPreps Community guidelines <https://www.nipreps.org/community/>`__.
Please, make sure you have read and understood all the documentation
provided in the `NiPreps portal <https://www.nipreps.org>`__ before
you get started.

Setting up your development environment
---------------------------------------
We believe that *fMRIPrep* must be free to use, inspect, and critique.
Correspondingly, you should be free to modify our software to improve it
or adapt it to new use cases and we especially welcome contributions to
improve it or its documentation.

We actively direct efforts into making the scrutiny and improvement processes
as easy as possible.
As part of such efforts, we maintain some
`tips and guidelines for developers <https://www.nipreps.org/devs/devenv/>`__
to help minimize your burden if you want to modify the software.

Internal configuration system
-----------------------------

.. automodule:: fmriprep.config
   :members: from_dict, load, get, dumps, to_filename, init_spaces

Workflows
---------

.. automodule:: fmriprep.workflows.base
.. automodule:: fmriprep.workflows.bold
.. automodule:: fmriprep.workflows.bold.fit
