import setuptools


with open("README.md", "r") as f, open("requirements.txt", "r") as g:
    long_description = f.read()
    required = g.read().splitlines()


setuptools.setup(
    name='photosynthesis_metrics',
    version='0.4.0',
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
