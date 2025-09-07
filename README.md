# MagCUSPS — Magnetopause Continuous Unsupervised Simulation Profiling & Synthesis

## Introduction

This library has been created in the context of a Master's Thesis in Computer science at Imperial College London in collaboration with the Space Plasma science research team. 

The first goal was to provide tools to extract certain topological elements from 3D voxel grids of simulation data extracted from ICL's Gorgon model. 
It provides optimized C++ tools to read and process data, obtain stream and field lines, and extract the magnetopause with less than a second of computation even with high precision.
Some other python tools are present for plotting and graphing purposes, as well as to provide extra tools to extract features like the current sheet and the X-line.

The main application of this library for computational models of the Earth magnetosphere is to determine in real time, through the extraction of the mentioned features, if the model is performing correctly or if it has failed in some way. 
Through statistical testing, it has been determined that the real time analysis is performant enough to be able to evaluate the output of the model with reasonable certainty.

Though this library has been created with the Gorgon model in mind, it should be model agnostic if the user provides their own implementation of the ReaderWriter interface present to read the output of their model into the provided Matrix class.    

## Installation

### Dependencies

The library provides an example implementation of a `.pvtr` to `.bin` ReaderWriter used for all of the tests. 
In the case that the user wants to test their installation with the provided tests or wants to use the `.pvtr` reader provided, they will need to have a installation of the C++ library: 
- **VTK**: can be obtained from [gitlab](https://gitlab.kitware.com/vtk/vtk) (only for the C++ library)

For the least squares fitting to the analytical models, the following dependencies will be installed:
- **Eigen** from [gitlab](https://gitlab.com/libeigen/eigen)
- **Abseil** from [github](https://github.com/abseil/abseil-cpp)
- **GoogleTest** from [github](https://github.com/google/googletest)
- **Ceres** from [github](https://github.com/ceres-solver/ceres-solver)

### C++ library

```bash
mkdir build && cd build
cmake ..
make
```

### Python library

Can be installed on any machine without needing to pull the repo:

```bash
pip install mag-cusps
```

Which can then be imported in python with:

```python
import mag_cusps as cusps
```

## Using the python library

Here is a complete example pipeline using the python library

```
import numpy as np
import mag_cusps as cusps

# define J, Rho, X, Y, Z, earth_pos from desired simulation

J_processed: np.ndarray = cusps.preprocess( J, X, Y, Z, new_shape )
Rho_processed: np.ndarray = cusps.preprocess( Rho, X, Y, Z, new_shape )
J_norm_processed = np.linalg.norm( J_processed, axis=3 )

BS = cusps.get_bowshock( Rho_processed, ... )
MP = cusps.get_interest_points( J_norm_processed, Rho_processed, ... )

MP_params, MP_cost = cusps.fit_to_analytical( MP, ... )

analytics = cusps.analyse( ... )
model = cusps.load_pretrained_model( ... )
quality_score = model.predict(analytics)                ### <- the quality scored associated with the input data
uncertainty = model.get_sample_uncertainty(analytics)   ### <- the uncertainty associated with that score
```



## Master's Thesis

### Abstract

The magnetosphere and its surrounding structures are critical in understanding space-weather dynamics. Many space plasma labs, including Imperial College London space plasma team, have created numerical simulations to forecast its evolution through time. This thesis presents a comprehensive framework that extracts the critical information from these numerical simulations in real time, to be able to preserve their important features through time, without the immense memory complexity normally associated with storing this data.
A topological analysis approach is developed that extracts magnetopause and bow-shock positions by identifying maxima in current density magnitude and minima in density gradient magnitude using a probabilistic search algorithm. This method, an improvement of the one introduced in Nemecek et al. 2011, reduces computational complexity by several orders of magnitude compared to storing full 3D simulation grids, compressing data from hundreds of megabytes to kilobytes per time-step while maintaining spatial accuracy. The results can be extracted as point grids or analytical function approximations. Significant improvements are provided in this area, presenting a new improved version based on the function described in Liu et al. 2012, including eccentricity to better emulate the possible shapes of the magnetopause, but also improving the cusps to now satisfy both C0 and C1 continuity.
From these results, the library provides means of evaluating in real time the quality of the numerical simulation data, and determine its breaking points. This is done using Random Forest regression, introduced in AAAAAAA, achieving R2 scores up to AAAAAA depending on the analytical model, successfully identifying erroneous data with recall rates up to AAAAAA, while limiting false alarms, for real time restarting possibilities for diverging simulations.


### Citation

To cite this work, please use ...

...
