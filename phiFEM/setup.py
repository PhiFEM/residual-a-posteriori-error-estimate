from setuptools import setup, find_packages

setup(
    name='phiFEM',
    version='0.1',
    packages=find_packages(),
    package_data={'phiFEM': ['py.typed']},
    include_package_data=True,
)