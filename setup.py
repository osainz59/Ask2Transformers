import setuptools
import a2t

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="a2t",
    version=a2t.__version__,
    author="Oscar Sainz",
    author_email="oscar.sainz@ehu.eus",
    description="Ask2Transformers is a library for zero-shot classification based on Transformers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/osainz59/Ask2Transformers",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["transformers", "tqdm", "torch", "numpy", "scikit-learn"],
)
