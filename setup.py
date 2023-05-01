from setuptools import setup, find_packages
import os


# extract version
with open(os.path.join(os.path.dirname(__file__),
          "fides", "version.py")) as f:
    version = f.read().split('\n')[0].split('=')[-1].strip(' ').strip('"')


# read a file
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# project metadata
setup(name='fides',
      version=version,
      description="python Trust Region Optimization",
      long_description=read('README.md'),
      long_description_content_type="text/markdown",
      author="Fabian Froehlich",
      author_email="froehlichfab@gmail.com",
      url="https://github.com/Fides-dev/fides",
      license="BSD-3-Clause",
      packages=find_packages(exclude=["doc*", "test*"]),
      install_requires=['numpy>=1.19.2',
                        'scipy>=1.5.2',
                        'h5py>=3.5.0'],
      tests_require=['pytest>=5.4.2',
                     'flake8>=3.7.2'],
      extras_require={},
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Science/Research',
                   'Topic :: Software Development :: Libraries',
                   'License :: OSI Approved :: BSD License',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3 :: Only',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10',
                   'Programming Language :: Python :: 3.11'],
      python_requires='>=3.9')
