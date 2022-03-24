# NNHelferlein.jl
Collection of little helpers to simplify various Machine Learning tasks
(esp. building neural networks with Knet).

The German word *Helferlein* means something like *little helper*;
please pronounce it like `hell-fur-line`.

See documentation and examples for a first intro.

<!---
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://andreasdominik.github.io/NNHelferlein.jl/stable)
--->
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://andreasdominik.github.io/NNHelferlein.jl/dev)
<!--
CI badge
[![Build Status](https://travis-ci.org/andreasdominik/NNHelferlein.jl.svg?branch=main)](https://travis-ci.org/andreasdominik/NNHelferlein.jl)
-->
[![Tests](https://github.com/andreasdominik/NNHelferlein.jl/actions/workflows/run_tests.yml/badge.svg)](https://github.com/andreasdominik/NNHelferlein.jl/actions/workflows/run_tests.yml) [![codecov](https://codecov.io/gh/andreasdominik/NNHelferlein.jl/branch/main/graph/badge.svg?token=9R12TMSKP1)](https://codecov.io/gh/andreasdominik/NNHelferlein.jl)

<!---
[![codecov.io](http://codecov.io/github/andreasdominik/NNHelferlein.jl/coverage.svg?branch=master)](http://codecov.io/github/andreasdominik/NNHelferlein.jl?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/LiScI-Lab/SOM.jl/badge.svg?branch=master)](https://coveralls.io/github/LiScI-Lab/SOM.jl?branch=master)
--->


# Installation

The package is not yet released but can be installed manually with the Julia
package manager.

```Julia
using Pkg
Pkg.add(url="https://github.com/andreasdominik/NNHelferlein.jl.git")
```

<!---
Due to a backwards incompatibility with the dependency `AutoGrad.jl`, it is
currently necessary to manually install the latest version of AutoGrad.jl instead
of the released version 1.2.4 to be used with NNHelferlein:

```Julia
using Pkg
Pkg.add(url="https://github.com/andreasdominik/NNHelferlein.jl.git")
Pgk.add(url="https://github.com/denizyuret/AutoGrad.jl.git")
```
--->


# Caveat:
Please be aware that the package is still in development and
not yet compltetely tested. You may already use it on own risk.

While reading this, I must add: the package is *almost* ready with an
not-so-bad test coverage. If you see the tests passing in the moment, 
it may be save to use the helpers.

As soon as dev. and tests are completed the package will be
registered (however maybe under a different name).
