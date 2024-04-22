from setuptools import setup, find_packages

setup(
    name='tts-with-rvc',
    version='0.1.2',
    description='TTS with RVC pipeline',
    author='Atm4x',
    package_dir={'': 'src'},
    packages=find_packages(),
    install_requires=[
        "huggingface_hub",
        "torch",
        "torchaudio",
        "edge-tts"
    ]
)
