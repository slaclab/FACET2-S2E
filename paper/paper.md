---
title: 'FACET2-S2E: Start-to-end simulations of the FACET-II beamline'
tags:
  - accelerator physics
  - plasma physics
  - python
authors:
  - name: Nathan Majernik
    corresponding: true
    orcid: 0000-0001-9977-0248
    affiliation: 1
  - name: Frederick Cropp
    orcid: 0000-0003-4473-0374
    affiliation: 1
  - name: Claudio Emma
    affiliation: 1
  - name: Thamine Dalichaouch
    affiliation: 2             
affiliations:
 - name: SLAC National Accelerator Laboratory, USA
   index: 1
 - name: University of California, Los Angeles, USA
   index: 2
date: 30 June 2025
bibliography: paper.bib
csl: apa.csl
journal: JOSS
---




# Summary

`FACET2-S2E` is a Python package for start-to-end simulations of the Facility for Advanced Accelerator Experimental Tests-II (FACET-II) [@yakimenko:2019], a US Department of Energy National User Facility. A kilometer-long particle accelerator creates, manipulates, and accelerates electron beams to over 10 GeV before focusing and compressing them to the micron-scale. These beams create extreme electric and magnetic fields on the femtosecond timescale, uniquely enabling research into exotic states and advanced accelerator technology, including plasma wakefield acceleration. This software package enables present or prospective facility users to easily run the most common types of simulation pipelines to design experiments and interpret results.


# Statement of need
The complexity of the underlying beam dynamics in particle accelerators generally necessitates that operators, users, and physicists who control, interpret accelerator data and design experiments, respectively, use simulations.  Sources of complexity include the downstream collective effects, such as coherent synchrotron radiation, and the electron source physics. In the case of FACET-II and other accelerators employing photoinjectors, the photocathode physics and the space-charge dominated physics in the electron gun is also an important consideration. FACET-II is a National User Facility focusing on the generation of ultra-bright, extremely dense electron bunches to serve a broad set of user experiments in applications ranging from strong-field QED, laboratory astrophysics, plasma wakefield acceleration, and x-ray science. Recent results include the generation of 100 kiloampere class electron bunches using laser based beam shaping techniques [@PhysRevLett.134.085001] and the demonstration of beam brightness transformation using a multi-stage plasma wakefield acceleration mechanism [@zhang2025].  The facility presents specific challenges from a simulation point of view as typical experiments generally use different simulation codes to model the beam generation, transport through the linear accelerator including beam shaping with lasers and collimators, and plasma simulations. 

One challenge for modeling complex systems is handling multiple simulation results in a single workflow.  For workflows using multiple simulation codes, one must ensure compatible handoffs between simulations.  For FACET-II, for reasons described in the Software Design section, three (3) different simulation codes are used.  The core workflow uses `IMPACT-T` [@qiang:2006] for beam generation and low energy transport, `Bmad`, `Tao`, and `PyTao` [@Sagan:Bmad2006] for most of the beam transport through the kilometer-long linear accelerator, and, optionally, `QPAD` [@li2021quasi] for particle-in-cell simulations of the beam in plasma, and `openPMD-beamphysics` [@christopher_mayes_2025_15477845] for handling beam files. 

`FACET2-S2E` is a Python package that abstracts away these challenges from the end-user.  It contains utilities, analysis notebooks, and configuration files used to perform start-to-end (S2E) simulations of the FACET-II particle accelerator beamline, a facility which hosts hundreds of external users a year. Users can quickly and easily run the most common types of simulations including parameter scans, constrained optimization of both Twiss and multiparticle tracking objectives, and jitter sensitivity analysis. The package seamlessly chains together multiple codes, provides reference configurations, offers convenience functions which mimic real-world feedback systems, and offers templates for common development and optimization tasks, so beam physicists can focus on the physics.

Using simulations from the control room for operators, users and physicists is a standard use-case.  FACET-II, like many accelerators, uses EPICS [@dalesio1991epics].  Another challenge of performing control room simulations is translating between the language and units of the underlying codes and the real-world EPICS control system.  This package provides an easy and convenient way to do simulations in the control room.  In fact, this package is readily integrated into control room tools, both importing real values into simulations and using the simulation optimizers to write new values to the machine. It also abstracts away details (e.g. specific magnet values) instead allowing users to focus on specifying desired higher-level goals (e.g. beam qualities).

# State of the field

Other software packages exist which simulate different parts of a beamline, for example `GPT` [@de1996general] and `IMPACT` [@qiang:2006] for low energy portions; `elegant` [@borland2000elegant] and  `Bmad` [@Sagan:Bmad2006] for higher energy regions; or `HiPACE++` [@diederichs2022hipace] and `QPAD` [@li2021quasi] for particle-in-cell beam-plasma interactions. These are general purpose codes which allow the design of arbitrary beamlines but their learning curves are commensurately steep. Further, they have different domains, requiring handoffs of the beam between codes. 

The standardization of simulations is a burgeoning sub-field within accelerator modeling, with the OpenPMD standard [@huebl_2015_33624] defining a standard beam distribution format, and the in-development Particle Accelerator Lattice Standard (PALS) [@pals_docs_2025] defining simulation and experimental accelerator lattice files.  Other packages aim to link together multiple codes, like LUME [@LUME]. However, this is still a general package. 

In contrast, `FACET2-S2E` is a bespoke package which models, using several underlying codes, a single beamline which is thoughtfully compartmentalized and abstracted, making it as frictionless as possible for members of the FACET-II user community to perform beamline simulations.

# Software design
`FACET2-S2E` wraps established accelerator simulation codes into a single workflow for the FACET-II beamline. `IMPACT-T` [@qiang:2006] generates the initial beam, `Bmad`, `Tao`, and `PyTao` [@Sagan:Bmad2006] handle the kilometre‑long transport and, optionally, `QPAD` [@li2021quasi] simulates plasma wakefield sections. Impact-T is a good choice for modeling photoinjectors and the electron source physics.  Bmad was chosen as a fast, but accurate solver outside that regime.  QPAD was chosen as a fast, highly efficient plasma simulation tool due to its hybrid algorithm which combines both the quasi-3D Fourier mode decomposition [@lifschitz09] and the quasi-static approximation [@mora97]. The quasi-3D algorithm decomposes fields and sources into azimuthal fourier modes on a cylindrical grid, thereby reducing the algorithmic complexity to that of a 2D code while still resolving 3D asymmetries. Under the quasi-static approximation, the particle beam evolves slowly over the plasma response period, thereby permitting large timesteps and orders of magnitude computational speedup over conventional PIC codes.
Beam files are exchanged via the openPMD format. Users interact with high‑level functions that abstract away the multiple underlying codes, as well as translating the units and language to match that of the EPICS variable names and units present in the control room. Configuration files centralise facility‑specific defaults and can be overridden without touching the core code, while helpers support parameter scans, optimization, and jitter studies. The result is a simplified toolkit tailored to a single beamline.

# Research impact statement

While under development, this package has been used to design experiments, support real-time execution of experiments from the control room, and aid in the interpretation of experimental results. Such experiments include those published as [@PhysRevLett.134.085001] and [@zhang2025].

# AI usage disclosure

Generative AI, specifically LLMs from OpenAI and Anthropic, were used in software and documentation creation. All AI generated content has been reviewed by humans before being included in the package.

# Acknowledgements

This work was supported by the U.S. Department of Energy under DOE Contract No. DEAC02-76SF00515 and the Consortium for Advanced Modeling of Particle Accelerators (CAMPA) funded under DOE Contract No. DE-AC02-05CH11231. The authors thank M. Ehrlichman and C. Mayes for sharing their Bmad expertise and D. Cesar for his floorplan plotting function.

# References
