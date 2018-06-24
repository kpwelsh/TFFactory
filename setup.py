import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TFFactory",
    version="0.1.3",
    author="Kevin Welsh",
    author_email="kevinwelsh132@gmail.com",
    description="A mock tensorflow package that constructs JSON objects instead of tensors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kevdog25/TFFactory",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)