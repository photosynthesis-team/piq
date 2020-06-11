import os
import codecs
import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]

    raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as f, open("requirements.txt", "r") as g:
    long_description = f.read()
    required = g.read().splitlines()

package_name = 'photosynthesis_metrics'
setuptools.setup(
    name=package_name,
    version=get_version(os.path.join(package_name, '__init__.py')),
    author="Sergey Kastryulin",
    author_email="snk4tr@gmail.com",
    description="Measures and metrics for image2image tasks. PyTorch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/photosynthesis-team/photosynthesis.metrics",
    install_requires=required,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
