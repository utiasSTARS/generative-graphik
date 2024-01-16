from setuptools import setup, find_packages

# TODO: see https://github.com/pymanopt/pymanopt/blob/master/setup.py for mmore later
setup(
    name="generative-graphik",
    version="0.1",
    description="Generative inverse kinematics",
    author="Filip Maric, Oliver Limoyo",
    author_email="filip.maric@robotics.utias.utoronto.ca, oliver.limoyo@robotics.utias.utoronto.ca",
    license="MIT",
    url="https://github.com/utiasSTARS/generative-graphik",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "urdfpy",
        "numpy >= 1.16",
        "liegroups @ git+ssh://git@github.com/utiasSTARS/liegroups@generative_ik#egg=liegroups",
        "graphIK @ git+ssh://git@github.com/utiasSTARS/graphIK@generative_ik#egg=graphIK",
        "networkx >= 2.8.7",
        "tensorboard"
    ],
    python_requires=">=3.8",
)
