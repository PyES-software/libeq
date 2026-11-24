project('libeq', 'c', 'fortran')

py = import('python').find_installation(pure: false)
numpy = dependency('numpy')
lapack = dependency('lapack')

f2py_dir = run_command(
  py, '-c', 'import numpy.f2py; print(numpy.f2py.get_include())',
  check: true,
).stdout().strip()

wrapper = custom_target(
  'fsolver-wrapper',
  input: 'src/libeq/solver/fsolver.pyf',
  output: ['fsolvermodule.c', 'fsolver-f2pywrappers2.f90'],
  command: [py, '-m', 'numpy.f2py', '@INPUT@'],
)

py.extension_module(
  'fsolver',
  [
    wrapper,
    f2py_dir / 'fortranobject.c',
    'src/libeq/solver/cpluss.f90',
    'src/libeq/solver/mb.f90',
    'src/libeq/solver/si.f90',
    'src/libeq/solver/err.f90',
    'src/libeq/solver/nr.f90',
    'src/libeq/solver/pcf.f90',
    'src/libeq/solver/outer.f90',
    'src/libeq/solver/eq.f90',
  ],
  include_directories: include_directories(f2py_dir),
  dependencies: [numpy, lapack],
  subdir: 'libeq/solver',
  install: true,
)

install_subdir('src/libeq', install_dir: py.get_install_dir())
