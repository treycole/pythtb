#!/usr/bin/env python

from distutils.core import setup

setup(name='pythtb',
      version='1.7.2',
      author='Sinisa Coh and David Vanderbilt',
      author_email='sinisacoh@gmail.com  dhv@physics.rutgers.edu',
      url='http://www.physics.rutgers.edu/pythtb',
      download_url='http://www.physics.rutgers.edu/pythtb',
      keywords='tight binding, solid state physics, condensed matter physics, materials science',
      py_modules=['pythtb'],
      license="gpl-3.0",
      description="Simple solver for tight binding models for use in condensed matter physics and materials science.",
      long_description="The tight binding method is an approximate approach for solving for the electronic wave functions for electrons in solids assuming a basis of localized atomic-like orbitals.",
      platforms=['UNIX','MAC OS X','Windows'],
      install_requires=['numpy','matplotlib'],
      )

