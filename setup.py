from setuptools import setup, find_packages

setup(
    name='dprox',
    packages=find_packages(),
    version='0.0.2',
    include_package_data=True,
    requires=['imageio','scikit_image', 'matplotlib', 'munch', 'tfpnp']
)
