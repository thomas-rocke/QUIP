from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sysconfig
import re
import os
import shutil

major_version = '0.9'

# source: https://peps.python.org/pep-0440/
# adapted for local version identifier
def is_canonical(version):
    return re.match(r'^([1-9][0-9]*!)?'
                    r'(0|[1-9][0-9]*)'
                    r'(\.(0|[1-9][0-9]*))*'
                    r'((a|b|rc)(0|[1-9][0-9]*))?'
                    r'(\.post(0|[1-9][0-9]*))?'
                    r'(\.dev(0|[1-9][0-9]*))?'
                    r'(\+[a-zA-Z0-9][a-zA-Z0-9\.]*)?$',
                    version) is not None

with open('VERSION') as fin:
    version_string = fin.readline().strip()
    if version_string.startswith('v'):
        version_string = version_string[1:]

    if is_canonical(version_string):
        version = version_string
    else:
        version_string = version_string.replace('-', '.')
        version = major_version + '+git' + version_string
        if not is_canonical(version):
            print(f"Warning: Version '{version}' is not PEP440 canonical. "
                  f"This may prevent a successful build of wheels.")

print('version:', version)

platform = sysconfig.get_platform() + "-" + sysconfig.get_python_version()
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# binary exectuables to bundle with distribution
package_data_files = ['quip', 'libquip.a']
# only add these if the executable actually exists
entry_points = ['quip=quippy.cli:quip',
                'quip-config=quippy.cli:quip_config']
if os.path.exists('gap_fit'):
    package_data_files.append('gap_fit')
    entry_points.append('gap_fit=quippy.cli:gap_fit')
if os.path.exists('md'):
    package_data_files.append('md')
    entry_points.append('md=quippy.cli:md')
if os.path.exists('vasp_driver'):
    package_data_files.append('vasp_driver')
    entry_points.append('vasp_driver=quippy.cli:vasp_driver')

class my_build_ext(build_ext):
    def build_extension(self, ext):
        if not os.path.exists(os.path.dirname(self.get_ext_fullpath(ext.name))):
            os.makedirs(os.path.dirname(self.get_ext_fullpath(ext.name)))
        shutil.copyfile(os.path.join(this_directory, f'quippy/_quippy{ext_suffix}'), self.get_ext_fullpath(ext.name))

setup(
    name='quippy-ase',
    version=version,
    maintainer='James Kermode',
    maintainer_email='james.kermode@gmail.com',
    description = 'ASE-compatible Python bindings for the QUIP and GAP codes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',

        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'License :: Public Domain',
        'License :: Other/Proprietary License',

        'Programming Language :: Fortran',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    url='https://github.com/libAtoms/QUIP',
    install_requires=['numpy>=1.13', 'f90wrap>=0.2.6', 'ase>=3.17.0'],
    python_requires=">=3.6",
    packages=['quippy'],
    package_data={'quippy': package_data_files},
    cmdclass={'build_ext': my_build_ext },
    ext_modules=[Extension('quippy._quippy', [])],
    entry_points={
        'console_scripts': entry_points
    }
)
