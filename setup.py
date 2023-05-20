from setuptools import setup, find_packages

setup(
    name='dprox',
    packages=find_packages(),
    version='0.1.0',
    include_package_data=True,
    requires=['imageio', 'scikit_image', 'matplotlib', 'munch', 'tfpnp']
)
