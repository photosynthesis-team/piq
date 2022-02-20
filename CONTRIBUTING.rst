==================
Contribution guide
==================


Any contributions you make are **greatly appreciated**. If you plan to:

* Fix a bug, please do so without any further discussion;
* Close one of `open issues <https://github.com/photosynthesis-team/piq/issues>`__, please do so if no one has been
  assigned to it;
* Contribute new features, utility functions or extensions, please create
  `GitHub Issue <https://github.com/photosynthesis-team/piq/issues/new/choose>`__ and discuss your idea.


Style
-----

We follow `Google Python Style Guide <http://google.github.io/styleguide/pyguide.html>`_ for code,
`Google Python Style Docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
for documentation and `commitizen style <https://github.com/commitizen/cz-cli>`_ for commit messages.

We ran `flake-8 <http://flake8.pycqa.org/en/latest/>`_\ , `mypy <https://mypy.readthedocs.io/en/stable/index.html>`_\ ,
`pytest <https://docs.pytest.org/en/stable/>`_ and `SonarCloud <https://sonarcloud.io>`_ during CI pipeline
to enforce those requirements.

You can run them locally in project folder:

#. ``flake-8``

   .. code-block:: bash

       pip install flake8
       flake8 .

#. ``pytest`` (additional libraries are used during checks)

   .. code-block:: bash

       pip install pytest tensorflow libsvm pybrisque scikit-image pandas tqdm
       pytest -x tests/

#. ``mypy``

   .. code-block:: bash

       python3 -m pip install mypy
       python3 -m mypy piq/ --allow-redefinition

#. ``pre-commit``

  .. code-block:: bash

      python3 -m pip install pre-commit
      pre-commit install

Developing PIQ
--------------

Contributions are what make the open source community such an amazing place to learn, inspire, and create.

#. Fork the Project
#. Create your Feature Branch (\ ``git checkout -b feature/AmazingFeature``\ )
#. Commit your Changes (\ ``git commit -m 'Add some AmazingFeature'``\ )
#. Push to the Branch (\ ``git push origin feature/AmazingFeature``\ )
#. Open a Pull Request
#. Get your PR reviewed, polished and approved
#. Enjoy making a good open source project even better :wink:

Documentation
-------------

PIQ uses `reStructuredText <https://docutils.sourceforge.io/docs/user/rst/quickref.html>`_
file format to write the documentation, `Sphinx <https://www.sphinx-doc.org/en/master/>`_
to build it and `readthedocs <https://readthedocs.org>`_ to host it.


To build the documentation locally:

#. Install PIQ

#. Install prerequisites

   .. code-block:: bash

       cd docs
       pip install -r requirements.txt

#. Generate the documentation HTML files. The generated files will be in ``docs/build/html``.

   .. code-block:: bash

       make html

#. Preview changes in your web browser.

   .. code-block:: bash

       open your_piq_folder/docs/build/html/index.html

When changing documentation (adding formulas, tables, etc.), **provide a rendered version of the result**
as part of your PR (e.g. add screenshot). Limit line length in docstrings to 80 characters, so that it fits into
Jupyter documentation popups.

Get in Touch
------------

Feel free to reach out to `one of maintainers <https://github.com/photosynthesis-team/piq#contacts>`_
if you have any questions.
