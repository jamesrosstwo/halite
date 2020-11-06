from setuptools import setup
from src.constants import SETTINGS

version = SETTINGS["gym"]["version"]

setup(name='halite_gym',
      version=version,
      install_requires=['gym']
)