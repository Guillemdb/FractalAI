from setuptools import setup, find_packages


setup(
    name="fractalai",
    description="FractalAI - swarm intelligence for the masses.",
    version="0.0.1",
    license="Apache 2.0",
    author="Guillem Duran",
    author_email="guillem.db@gmail.com",
    url="https://github.com/FragileTheory/FractalAI",
    download_url='https://github.com/FragileTheory/FractalAI',
    packages=find_packages(exclude=("fractalai.tests",)),
    keywords=["fractalai", "reinforcement learning", "artificial intelligence",
              "monte carlo", "automaton-based"],
    install_requires=["numpy", "gym[atari]"],
    package_data={"": ["LICENSE", "README.md"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries"
    ]
)
