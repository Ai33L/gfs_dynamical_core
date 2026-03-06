# Initial Concept
Home to the GFS dynamical core, extracted from CliMT for standalone use.

# Product Guide: gfs-dynamical-core

## Vision
To provide a modular, easy-to-use, and highly accessible Python interface for the GFS dynamical core, enabling educators and students to explore atmospheric dynamics within the Sympl and CliMT ecosystems.

## Target Audience
- **Educators and Students**: Our primary focus is on making atmospheric modeling accessible for teaching and learning.
- **Atmospheric Scientists**: Researchers looking for a lightweight, standalone dynamical core for experimental setups.

## Key Goals
- **Ease of Use & Integration**: Prioritize a clean Pythonic API that hides the complexity of the underlying Fortran/Cython implementation.
- **Modularity & Decoupling**: Maintain the dynamical core as a strictly standalone component that can be easily plugged into different modeling frameworks.
- **Compatibility**: Ensure seamless integration with `sympl` and `climt`.

## Core Features
- **Sympl/CliMT Integration**: Full support for the Sympl data model and CliMT component architecture.
- **Standalone Dynamical Core**: Independent build and execution environment for the GFS dynamics.
- **JAX Acceleration**: A differentiable, pure functional JAX implementation of the dynamical core.
- **Hardware Agnostic**: Support for CPU, GPU, and TPU execution via JAX.

## Future Roadmap
- **Modernization**: Gradually refactor and modernize the legacy code to improve maintainability and performance.
- **Accessibility & Documentation**: Develop comprehensive documentation, tutorials, and examples tailored for educational use.