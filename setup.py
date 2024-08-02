"""mcfacts

Long description goes here...
"""
from datetime import date
from setuptools import find_packages, setup, Extension
import os

#-------------------------------------------------------------------------------
#   Version
#-------------------------------------------------------------------------------
VERSIONFILE="__version__.py"
with open(VERSIONFILE, 'r') as F:
    _line = F.read()
__version__  = _line.split("=")[-1].lstrip(" '").rstrip(" '\n")
print(__version__)

#-------------------------------------------------------------------------------
#   GENERAL
#-------------------------------------------------------------------------------
__name__        = "mcfacts"
__date__        = date(2024, 4, 23)
__keywords__    = [
    "astronomy",
    "information analysis",
    "machine learning",
    "physics",
]
__status__      = "Alpha"


#-------------------------------------------------------------------------------
#   URLS
#-------------------------------------------------------------------------------
__url__         = "https://github.com/bmckernan/test_mcfacts"
__bugtrack_url__= "https://github.com/bmckernan/test_mcfacts/issues"


#-------------------------------------------------------------------------------
#   PEOPLE
#-------------------------------------------------------------------------------
__author__      = "Barry McKernan, Saavik Ford, Harry Cook, Vera Delfavero, Richard O'Shaughnessy"
__author_email__= "" #TODO

__maintainer__      = "Barry McKernan, Saavik Ford, Harry Cook, Vera Delfavero, Richard O'Shaughnessy"
__maintainer_email__= "" #TODO

__credits__     = ("Barry McKernan, Saavik Ford, Harry Cook, Vera Delfavero, Richard O'Shaughnessy",)


#-------------------------------------------------------------------------------
#   LEGAL
#-------------------------------------------------------------------------------
__copyright__   = 'Copyright (c) 2024 {author} <{email}>'.format(
    author=__author__,
    email=__author_email__
)

# TODO pick a license
__license__     = 'MIT License'
__license_full__= '''
MIT License

{copyright}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''.format(copyright=__copyright__).strip()


#-------------------------------------------------------------------------------
#   PACKAGE
#-------------------------------------------------------------------------------

DOCLINES = __doc__.split("\n")

CLASSIFIERS = [
"Development Status :: 3 - Alpha",
"Programming Language :: Python :: 3",
"Operating System :: OS Independent",
"Intended Audience :: Science/Research",
"Topic :: Scientific/Engineering :: Astronomy",
"Topic :: Scientific/Engineering :: Physics",
"Topic :: Scientific/Engineering :: Information Analysis",
]

# Matching the numpy version of the installation is a hacky fix for this bug:
# https://numpy.org/devdocs/user/troubleshooting-importerror.html#c-api-incompatibility
# If you have a beter solution please open an issue or pull request
REQUIREMENTS = {
    "install" : [
        "numpy>=1.23.1",
        "matplotlib>=3.5.2",
        "scipy>=1.11.2",
        "pandas",
    ],
    "setup" : [
        "pytest-runner",
    ],
    "tests" : [
        "pytest",
    ]
}

ENTRYPOINTS = {
    "console_scripts" : [
        # Example:
        # script_name = module.2.import:function_to_call
    ]
}


metadata = dict(
    name        =__name__,
    version     =__version__,
    description =DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    keywords    =__keywords__,

    author      =__author__,
    author_email=__author_email__,

    maintainer  =__maintainer__,
    maintainer_email=__maintainer_email__,

    url         =__url__,
#    download_url=__download_url__,

    license     =__license__,

    classifiers=CLASSIFIERS,

    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"": ['inputs/data/*.txt', 'vis/data/*.txt']},

    install_requires=REQUIREMENTS["install"],
    setup_requires=REQUIREMENTS["setup"],
    tests_require=REQUIREMENTS["tests"],

    entry_points=ENTRYPOINTS,
    #ext_modules=ext_modules,
    python_requires=">=3.10",
)

setup(**metadata)
