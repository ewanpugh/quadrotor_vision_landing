from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['quadrotor_vision_landing'],
    package_dir={'': 'src'}
)
setup(**d)
