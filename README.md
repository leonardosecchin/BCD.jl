# BCD

This is a Julia implementation of the Block Coordinate Descent (BCD) method with
quadratic regularization for minimizing functions with Hölder continuous
gradient, as described in

Amaral, Andreani, Secchin, Silva. Flexible block coordinate descent methods for
unconstrained optimization under Hölder continuity. 2026

## Installation and use

`]add https://github.com/leonardosecchin/BCD`

For usage instructions, type `?bcd` after load the package `BCD`. If HSL MA57 is
installed, it is used; otherwise, linear systems are solved by Julia's built-in
solver.

Please visit <https://github.com/JuliaSmoothOptimizers/HSL.jl> for instructions
on how install HSL MA57. For proper working, load HSL package before the BCD:

```
using HSL
using BCD
```

You can test if HSL is working by running `LIBHSL_isfunctional()`, which should
return true.


## Funding

This research has been partially supported by CEPID-CeMEAI (FAPESP 2013/07375-0),
FAPESP (grant 2023/08706-1), and National Council for Scientific and
Technological Development (CNPq) (grants 407147/2023-3, 306988/2021-6,
302520/2025-2, 401864/2022-7, and 306593/2022-0).


## How to cite

If you use this code in your publications, please cite us. For now, you can cite
the preprint:

Amaral, Andreani, Secchin, Silva. Flexible block coordinate descent methods for
unconstrained optimization under Hölder continuity. 2026
