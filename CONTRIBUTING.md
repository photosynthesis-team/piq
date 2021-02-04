# Contribution guide

### Issues

Use [GitHub Issues](https://github.com/photosynthesis-team/piq/issues) for bug reports and feature requests.


### Developing PIQ

Contributions are what make the open source community such an amazing place to learn, inspire, and create. 
Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
6. Get your PR reviewed, polished and approved
7. Enjoy making a good open source project even better :wink:

### Writing Documentation

PIQ uses [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) for formatting
docstrings.
Length of line inside docstrings block must be limited to 80 characters to fit into Jupyter documentation popups.

#### Building Documentation

To build the documentation locally:
1. Build and install PIQ
2. Install prerequisites
    ```bash
    cd docs
    pip install -r requirements.txt
    ```
3. Generate the documentation HTML files. The generated files will be in `docs/build/html`.
    ```bash
    cd docs
    make html
    ```
4. Preview changes in your web browser.
    ```bash
    open your_piq_folder/docs/build/html/index.html
    ```
#### Submitting changes for review

It is helpful when submitting a PR that changes the docs to provide a rendered version of the result. If your change is
small, you can add a screenshot of the changed docs to your PR.


### Code Style

Please follow [Google Python style guide](http://google.github.io/styleguide/pyguide.html) as a guidance on your code
style decisions. 
The code will be checked with [flake-8 linter](http://flake8.pycqa.org/en/latest/) during the CI pipeline. 
Standard PyCharm formatter and style checker appears to be more than enough in practice. 
Use [commitizen](https://github.com/commitizen/cz-cli) commit style where possible for simplification of understanding
of performed changes.

### Get in Touch

Feel free to reach out to [one of contributors](https://github.com/photosynthesis-team/piq#contacts)
if you have any questions.