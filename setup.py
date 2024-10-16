from setuptools import setup, find_packages


def load_requirements(filename="requirements.txt"):
    with open(filename, "r") as file:
        return [line.strip() for line in file.readlines()]


setup(
    name="pets",
    version="0.0.0",
    url="https://github.com/alec-tschantz/pets-jax",
    packages=find_packages(),
    install_requires=load_requirements(),
)
