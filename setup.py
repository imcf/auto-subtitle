from setuptools import find_packages, setup

setup(
    version="1.0",
    name="auto_subtitle",
    packages=find_packages(),
    py_modules=["auto_subtitle"],
    author="Miguel Piedrafita",
    install_requires=[
        "openai-whisper",
        "ffmpeg-python",
    ],
    extras_require={
        "gui": ["Flask"],
    },
    description="Automatically generate and embed subtitles into your videos",
    entry_points={
        "console_scripts": ["auto_subtitle=auto_subtitle.cli:main"],
    },
    include_package_data=True,
)
