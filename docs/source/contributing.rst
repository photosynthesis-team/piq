Contributing
============

The project follows `Google Python style guide <http://google.github.io/styleguide/pyguide.html>`_.
The code is checked with `flake-8 linter <http://flake8.pycqa.org/en/latest/>`_ during the CI pipeline.
The `commitizen <https://github.com/commitizen/cz-cli>`_ commit style is used for simplification of understanding of
performed changes.

Issues
^^^^^^
Use `GitHub Issues <https://github.com/photosynthesis-team/piq/issues>`_ for bug reports and feature requests.


Developing PIQ
^^^^^^^^^^^^^^
Contributions are what make the open source community such an amazing place to learn, inspire, and create.
Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (``git checkout -b feature/AmazingFeature``)
3. Commit your Changes (``git commit -m 'Add some AmazingFeature'``)
4. Push to the Branch (``git push origin feature/AmazingFeature``)
5. Open a Pull Request
6. Get your PR reviewed, polished and approved
7. Enjoy making a good open source project even better :wink:

Writing Documentation
^^^^^^^^^^^^^^^^^^^^^

PIQ uses `Google style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_ for formatting
docstrings.
Length of line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups.

Building Documentation
''''''''''''''''''''''

To build the documentation locally:

1. Build and install PIQ

2. Install prerequisites

.. code-block:: bash

    cd docs
    pip install -r requirements.txt

3. Generate the documentation HTML files. The generated files will be in `docs/build/html`.

.. code-block:: bash

    cd docs
    make html

4. Preview changes in your web browser.

.. code-block:: bash

    open your_piq_folder/docs/build/html/index.html

Submitting Changes for Review
'''''''''''''''''''''''''''''

It is helpful when submitting a PR that changes the docs to provide a rendered version of the result. If your change is
small, you can add a screenshot of the changed docs to your PR.


Get in Touch
^^^^^^^^^^^^

Feel free to reach out to `one of contributors <https://github.com/photosynthesis-team/piq#contacts>`_
if you have any questions.