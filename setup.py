import setuptools

__version__ = "0.2.0"

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="surrDAMH",
    version=__version__,
    license='GPL 3.0',
    description='Surrogate accelerated Markov chain Monte Carlo methods for Bayesian inversion,'
                'based on the delayed acceptance Metropolis-Hastings algorithm',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Simona Bérešová',
    author_email='simona.beresova@ugn.cas.cz',
    url='https://github.com/dom0015/surrDAMH',
    # download_url='https://github.com/dom0015/surrDAMH/archive/v{__version__}.tar.gz',
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],

    keywords=[
        'Bayesian inversion', 'Surrogate', 'Metropolis-Hastings', 'Markov chain', 'Monte Carlo',
    ],
    # include_package_data=True, # package includes all files of the package directory
    zip_safe=False,
    install_requires=['numpy>=1.13.4', 'scipy', 'pandas', 'matplotlib', 'mpi4py'],
    python_requires='>=3',

    packages=setuptools.find_packages(),
    # package_dir={
    #     '': '.',
    # },
    # package_data={
    #     "endorse" : ["*.txt"],
    #     "endorse.flow123d_inputs": ['*.yaml']
    # },
    # entry_points={
    #     'console_scripts': ['endorse_gui=endorse.gui.app:main', 'endorse_mlmc=endorse.scripts.endorse_mlmc:main']
    # }
)
