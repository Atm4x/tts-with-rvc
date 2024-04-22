from setuptools import setup, find_packages

setup(
    name='tts_with_rvc',
    version='0.1.2',
    description='TTS with RVC pipeline',
    author='Atm4x',
    package_dir={'': 'src'},  # Include this line to specify the source directory
    packages=find_packages(),  # Update find_packages to look in 'src' director
    install_requires=[
        "huggingface_hub",
        "torch",
        "torchaudio",
        "edge-tts"
    ],
    scripts=['inference.py']
)
