from setuptools import setup, find_packages

setup(
    name='tts-with-rvc',
    version='0.1.1',
    description='TTC with RVC pipeline',
    author='SunSual',
    packages=find_packages(),
    install_requires=[
        "huggingface_hub",
        "torch",
        "torchaudio",
        "edge-tts"
    ]
)