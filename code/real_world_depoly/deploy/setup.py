from setuptools import find_packages, setup


subpackages = find_packages(where='.', exclude=('go2_gym_deploy', 'go2_gym_deploy.*'))
packages = ['go2_gym_deploy'] + [
    f'go2_gym_deploy.{package}' for package in subpackages
]
package_dir = {'go2_gym_deploy': 'go2_gym_deploy'}
package_dir.update({
    f'go2_gym_deploy.{package}': package.replace('.', '/')
    for package in subpackages
})

setup(
    name='go2_gym_deploy',
    version='1.0.0',
    author='Gabriel Margolis',
    license="BSD-3-Clause",
    packages=packages,
    package_dir=package_dir,
    author_email='gmargo@mit.edu',
    description='Toolkit for deployment of sim-to-real RL on the Unitree Go2.'
)
