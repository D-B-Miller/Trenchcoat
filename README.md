# Plasma Spray iCoating

Thermal spray coating is a manufacturing process where a solid substrate is coated with a material by a plasma stream. A carrier gas is converted to plasma by a high-power electrode and has particles of a coating material introduced into the flow. The plasma carries the material to the substrate where it hopefully adheres to the surface. Possible coatings include paint, metal and polymers. The process needs to be monitored to ensure that the particles adhere, a smooth coating is applied and defects aren't introduced into the coating. 

Possible conclusions include 
  - The coating particles not being hot enough to adhere
  - Particles bouncing off the surface due to an obtuse collision angle
  - Particles being too slow to adhere

## Requirements

This is more of a general list of requirements but it is recommended to look at the individual folders for specific areas

### Hard Requirements

- [numpy](https://numpy.org/)
- [scipy](https://scipy.org/)
- [matplotlib](https://matplotlib.org/)
- [cv2](https://pypi.org/project/opencv-python/)
- [Jupyter Notebook](https://jupyter.org/)
- [h5py](https://www.h5py.org/)
- [ipympl](https://matplotlib.org/ipympl/)
    + For interacting with matplotlib in Jupyter notebooks
- [Exiftool](https://exiftool.org/)
    + Required for reading FLIR files
    + Needs to be downloaded and placed on PATH (for Windows)
    + The path to EXE can also be provided in Python wrapper
- [PyExiftool](https://pypi.org/project/PyExifTool/)
    + Python wrapper around exiftool for calling and processing the returned values

### Recommendations

- [VSCode](https://code.visualstudio.com/) or other IDE (makes it easier to interact with Jupyter notebooks)
    + Easier to install and set up Jupyter notebooks

## Project Structure

Multiple sensor modalities are investigated to sensorize the process and build a digital twin. This project is organised around the different modalities due to different data requirements.

- [scripts](src)
  + Program files for connecting to sensors, processing the data and generating results
  + [trenchcoat](src/trenchcoat)
      * Python API for processing the thermal data produced by a FLIR T540 and the Acoustic Emission Sensors
        
