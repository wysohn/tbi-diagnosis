from setuptools import setup

# https://stackoverflow.com/questions/26900328/install-dependencies-from-setup-py
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name='TBI Diagnosis',
    version='1.0',
    description='Diagnosis of TBI using Pulsatility Ultrasound',
    packages=['src'],
    install_requires=install_requires
)