from setuptools import setup, find_packages

setup(name='surrMCMC',
 
      version='0.1',
 
      url='https://github.com/tralalala/tralalala',
 
      license='tralalala',
 
      author='Lala Trala',
 
      author_email='lala.trala@gmail.com',
 
      description='MCMC sampling in Bayesian inversion accelerated by surrogate models',
 
      packages=find_packages(exclude=['tests']),
 
      long_description=open('README.md').read(),
 
      zip_safe=False,
 
#      setup_requires=['nose>=1.0'],
 
#      test_suite='nose.collector'
      
      )
