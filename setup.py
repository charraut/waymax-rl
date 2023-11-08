from setuptools import setup


# Function to open and read the file for the requirements
def read_requirements():
    with open("requirements.txt") as req:
        content = req.read()
        requirements = content.split("\n")

    return requirements


setup(
    name="waymax-rl",
    packages=["waymax_rl"],
    version="0.1",
    description="",
    python_requires=">=3.10",
)
