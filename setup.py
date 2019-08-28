import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nnplot",
    # packages = ['nnplot'],
    version="1.0.1",
    author="Yuval Ai",
    author_email="yuval_a@rad.com",
    description="Plot Neural Networks and the weights of the links between their nodes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Yuval-Ai/nnplot",
    keywords = ['nn', 'ai', 'visualizer', 'learning', 'artificial', 'intelligence','weights'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
