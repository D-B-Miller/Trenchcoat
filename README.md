# Plasma Spray iCoating

Thermal spray coating is a manufacturing process where a solid substrate is coated with a material by a plasma stream. A carrier gas is converted to plasma by a high-power electrode and has particles of a coating material introduced into the flow. The plasma carries the material to the substrate where it hopefully adheres to the surface. Possible coatings include paint, metal and polymers. The process needs to be monitored to ensure that the particles adhere, a smooth coating is applied and defects aren't introduced into the coating. 

Possible conclusions include 
  - The coating particles not being hot enough to adhere
  - Particles bouncing off the surface due to an obtuse collision angle
  - Particles being too slow to adhere

## External requirements

- [Exiftool](https://exiftool.org/)
    + Required for reading metadata FLIR files
    + Needed for PyExifTool
    + Needs to be downloaded and placed on PATH (for Windows)
    + The path to EXE can also be provided in Python wrapper

## Recommendations

- [VSCode](https://code.visualstudio.com/) or other IDE (makes it easier to interact with Jupyter notebooks)
    + Easier to install and set up Jupyter notebooks

## Project Structure

Multiple sensor modalities are investigated to sensorize the process and build a digital twin. This project is organised around the different modalities due to different data requirements.

- [scripts](src)
  + Program files for connecting to sensors, processing the data and generating results
  + [trenchcoat](src/trenchcoat)
      * Python API for processing the thermal data produced by a FLIR T540 and the Acoustic Emission Sensors
  + [examples](examples)
      * Examples of using the API
        
## Link to Data

## Link to Papers

## Examples
