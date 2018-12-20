from setuptools import setup


packages = [
    'pystim',
]
install_requires = [
    'pycairo',
    'matplotlib',
    'numpy',
    'pandas',
    'Pillow',
    'scipy',
]
entry_points = {
    'console_scripts': [
        'pystim = pystim.__main__:main'
    ]
}

setup(
    name='pystim',
    version='0.0.0',
    packages=packages,
    install_requires=install_requires,
    entry_points=entry_points,
)
