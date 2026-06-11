# Product Guidelines: gfs-dynamical-core

## Design Principles
- **Backend Agnosticism**: Implementation must be compatible with multiple array backends (e.g., NumPy, JAX, PyTorch) as supported by the latest `sympl` and `climt` versions.
- **Clarity over Cleverness**: Scientific code should be readable and explainable. Avoid overly complex abstractions that obscure the physical equations.
- **Fail Fast**: Validate inputs and physical constraints early and provide clear, backend-appropriate error messages.

## User Experience (UX)
- **Scientific Consistency**: Use standard physical units and follow naming conventions established in `sympl` and `climt`.
- **Portable Metadata**: Metadata handling should be robust across different array backends, avoiding hard dependencies on specific wrappers like `pint` or `xarray` in the core logic.
- **Educational Flow**: Examples and tutorials should demonstrate how to use the core with different backends where applicable.

## Writing & Documentation Style
- **Pedagogical Tone**: Write documentation assuming the reader is a student or researcher who may not be an expert in the underlying Fortran implementation.
- **Markdown First**: Use Markdown for documentation and inline comments for implementation details.
- **Physical Context**: Explicitly document the physical assumptions and equations being solved by each part of the core.

## Visual & Formatting Standards
- **Standard PEP 8**: Follow standard Python formatting for all user-facing code.
- **Backend-Native Visuals**: Ensure integration with visualization tools that support various backends.
