from setuptools import setup

setup(name='gamdist',
      version='0.1',
      description='Generalized Additive Models',
      url='http://www.gotinder.com',
      author='Bob Wilson, Tinder',
      author_email='bob.wilson@gotinder.com',
      license='Apache v2.0',
      packages=['gamdist'],
      install_requires=[
          'numpy',
          'scipy',
          'pickle',
          'multiprocessing',
          'matplotlib',
          'cvxpy',
          'math'
      ],
      zip_safe=False)
