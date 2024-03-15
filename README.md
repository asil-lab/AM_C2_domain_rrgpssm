# Learning in High-Dimensional Dynamical Systems Using Gaussian Process State-Space Model

Thus is a Python library for reproducing the work published in [Learning in High-Dimensional Dynamical Systems Using Gaussian Process State-Space Model](https://link-url-here.org).

## Description
There are 4 main folders:
* **./main:** contains the code for the simulations in the paper.
* **./output:** contains simulation output files used to generate the plots as shown in the paper. Format is **.npz**.
* **./plot:** contains the code for exactly recreating the plots in the paper.
* **./util:** contains additional code for simulations.

Key points to consider:
* The paths specified in the code are specified relative to the location of the code. Consider changing the path appropriately.
* The codes in /main folder can be run again with different simulation parameters. Be sure to change the save folder path if necessary.


## Support and questions to the community

Ask questions using the issues section.

## Supported Platforms:

[<img src="https://www.python.org/static/community_logos/python-logo-generic.svg" height=40px>](https://www.python.org/)
[<img src="https://upload.wikimedia.org/wikipedia/commons/5/5f/Windows_logo_-_2012.svg" height=40px>](http://www.microsoft.com/en-gb/windows)
[<img src="https://upload.wikimedia.org/wikipedia/commons/8/8e/OS_X-Logo.svg" height=40px>](http://www.apple.com/osx/)
[<img src="https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" height=40px>](https://en.wikipedia.org/wiki/List_of_Linux_distributions)

Python 3.5 and higher

## Citation

    @Misc{Mishra2024,
      author =   {{A. Mishra and R. T. Rajan}},
      title =    {{Learning in High-Dimensional Dynamical Systems Using Gaussian Process State-Space Model}},
      howpublished = {\url{https://link-url-here.org}},
      year = {2024}
    }

## Getting started:

The code is written in Python.

### Python packages:

Packages to be installed:

    python3-dev
    build-essential   
    scipy
    numpy

### Running simulations:

Lotka-Volterra model for Svensson et al (https://proceedings.mlr.press/v51/svensson16.html):

    python3 $DIR$/main/rrgpssm_lotka_volterra_svensson.py

Lotka-Volterra model with proposed algorithm making use of domain-specific knowledge:

    python3 $DIR$/main/rrgpssm_lotka_volterra_domain_knowledge.py

Target tracking application with proposed algorithm making use of domain-specific knowledge:

    python3 $DIR$/main/rrgpssm_target_tracking_domain_knowledge.py


## Funding Acknowledgements

* This work is partially funded by the European Leadership Joint Undertaking (ECSEL JU), under grant agreement No 876019, the ADACORSA project - ”Airborne Data Collection on Resilient System Architectures.”
