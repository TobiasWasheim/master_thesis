### MSc-thesis of Tobias Washeim
#### Title:
Exploring Non-Trivial Thermal Production Mechanisms for Dark Matter
#### Summary:
The standard freeze-out process for Dark Matter is typically modeled using a single, momentum-integrated
Boltzmann equation for the abundance. This approach makes several simplifying assumptions, one of which is
that kinetic decoupling occurs much later than chemical decoupling. This ensures that the Dark Matter species
remains in local thermodynamic equilibrium with the plasma, maintaining the same temperature.
However, in some scenarios, this assumption does not hold. For example, the Dark Matter particles may
kinetically decouple earlier, leading to deviations from local thermodynamic equilibrium. Such deviations can
significantly impact the final relic abundance and require a more detailed treatment involving momentumdependent
Boltzmann equations.
In this project, we will investigate these non-trivial thermal production mechanisms for Dark Matter using
both analytical approximations and numerical solutions. Starting from the foundational work in astroph/
9903034, we aim to:
1. Understand the effects of early kinetic decoupling on the Dark Matter abundance.
2. Explore scenarios where alternative thermal production mechanisms dominate.
3. Develop tools to solve momentum-dependent Boltzmann equations numerically.
### Tasks and Timeline:

1. **Literature:**
   - Read and summarize the key results from astro-ph/9903034.
   - Review related works, such as arXiv:1805.00526, focusing on the assumptions and limitations of
standard freeze-out models.
1. **Analytical Investigation:**
   - Derive the momentum-dependent Boltzmann equation for Dark Matter production, starting from
    the general collision term.
    - Identify regimes where analytical approximations are valid (e.g., early or late decoupling).
    - Explore corrections to the relic abundance from early kinetic decoupling.
2. **Numerical Implementation:**
    - Develop a Python code to solve the momentum-dependent Boltzmann equation.
    - Validate the code against known results for standard freeze-out.
    - Investigate the parameter space for non-trivial production mechanisms, varying cross-sections,
    masses, and initial conditions.
3. **Elsatic Scattering Studies**
    - Use non Maxwell-Boltzmann distributions and the developed Python code to show that when only considering elastic scattering that the distribution goes toward a Maxwell-Boltzmann distribution
4. **Case Studies**
    - Explore specific models where early kinetic decoupling is expected, such as:
      - Dark Matter with large self-interactions.
      - Non-minimal couplings to the Standard Model plasma.
    - Analyze the impact of these scenarios on the relic abundance and velocity distribution of Dark
    Matter.
5. **Comparison with Observables**
    - Compare the results of the numerical solutions with observational constraints, such as:
      - Cosmic Microwave Background (CMB) measurements.
      - Large-scale structure data.
      - Indirect detection limits.
6. **Thesis Writing**
    - Compile the analytical and numerical results into a coherent narrative.
    - Ensure all code and results are reproducible and well-documented.
7. **Thesis Submission:**
    - Submit the thesis by the official deadline.