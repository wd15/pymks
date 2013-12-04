{
 "metadata": {
  "name": ""
 },
 "nbformat": 3,
 "nbformat_minor": 0,
 "worksheets": [
  {
   "cells": [
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "# Introductory Tutorial to Calculate MKS Influence Coefficients in Reciprocal Space\n",
      "\n",
      "The following notebook is an introductory tutorial describing how to calculate influence coefficients in the MKS in both real and reciprocal space. The MKS relationship is given by,\n",
      "\n",
      "$ p_{a,i} = \\alpha_j^h m_{a,i + j}^h $\n",
      "\n",
      "where the $p_{a,i}$ are the responses, $\\alpha_i^h$ are the influence coefficients and $m_{a,i}^l$ are the discretized microstructure. The $i$ and $j$ indices run over the spatial discretization, the $a$ index runs over the number of samples and the $h$ index runs over the microstructure discretization. In this notebooks we will,\n",
      "\n",
      " * Create a response function using the Cahn-Hilliard equation.\n",
      " \n",
      " * Demonstrate Cahn-Hilliard evolution.\n",
      " \n",
      " * Create some sample data.\n",
      " \n",
      " * Solve MKS regression in real space for demonstration purposes.\n",
      " \n",
      " * Solve MKS regression in Fourier space (the meat and bones).\n",
      " \n",
      " * Test that MKS regression seems to work with a single test sample."
     ]
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Calculating the Response"
     ]
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "In the MKS a sample set of microstructures and responses are required. In this example we will use the Cahn-Hilliard equation to provide the example response. [FiPy](http://www.ctcms.nist.gov/fipy/) is used to solve the governing equation, which is given by,\n",
      "\n",
      "$ \\frac{\\partial \\phi}{\\partial t} = \\nabla \\cdot D \\nabla \\left( \\frac{\\partial f}{\\partial \\phi}   - \\epsilon^2 \\nabla^2 \\phi \\right).$\n",
      "\n",
      "where the free energy is given by,\n",
      "\n",
      "$ f = (a^2/2) \\phi^2 (1 - \\phi)^2 $\n",
      "\n",
      "In this example $D = 1$ and $a = 1$. See [the FiPy CH example](http://www.ctcms.nist.gov/fipy/examples/cahnHilliard/generated/examples.cahnHilliard.mesh2DCoupled.html) for further details.\n",
      "\n",
      "The `fipy_response` function takes an initial field, $\\phi_0$ (the microstructure), and returns $\\frac{\\partial \\phi}{\\partial t}$ (the response). \n"
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "import numpy as np\n",
      "import fipy as fp\n",
      "import pylab as plt\n",
      "\n",
      "np.random.seed(3985)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 1
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "def fipy_response(phi0, dt, N):\n",
      "    nx = ny = N\n",
      "    mesh = fp.PeriodicGrid2D(nx=nx, ny=ny, dx=0.25, dy=0.25)\n",
      "    phi = fp.CellVariable(name=r\"$\\phi$\", mesh=mesh, value=phi0.copy())\n",
      "    PHI = phi.arithmeticFaceValue\n",
      "    D = a = epsilon = 1.\n",
      "    eq = (fp.TransientTerm()\n",
      "      == fp.DiffusionTerm(coeff=D * a**2 * (1 - 6 * PHI * (1 - PHI)))\n",
      "      - fp.DiffusionTerm(coeff=(D, epsilon**2)))\n",
      "    \n",
      "    eq.solve(phi, dt=dt, solver=fp.LinearLUSolver())\n",
      "    \n",
      "    return (np.array(phi) - phi0) / dt\n",
      "    \n"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 2
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Deomonstrate Cahn-Hilliard Evolution"
     ]
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "The following cell iterates the `fipy_response` function to demonstrate the evolution of the microstructure for an initially uniform random field. Using the `fipy_response` function is quite an inefficient method of using [FiPy](http://www.ctcms.nist.gov/fipy/), but useful for these demonstration purposes."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "from IPython.display import clear_output\n",
      "import time\n",
      "\n",
      "N = 10\n",
      "phi0 = np.random.random(N * N)\n",
      "dt = 1e-3\n",
      "\n",
      "fig = plt.figure()\n",
      "\n",
      "for i in range(100):\n",
      "    response = fipy_response(phi0, dt=dt, N=N)\n",
      "    #Euler forward\n",
      "    phi0 = response * dt + phi0\n",
      "    #print phi0\n",
      "    plt.contourf(phi0.reshape((N,N)))\n",
      "    time.sleep(1)\n",
      "    clear_output()\n",
      "    display(fig)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "metadata": {},
       "output_type": "display_data",
       "png": "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\n7yuLnarX62Ofv5F70LVaTdVq9cg10FYvv7t69aoqlYrrMY5oNpu6dOmSarWaWq2Wms2m65HUbDbV\nbDZVrVbVarX0448/uh6py9qWWaPR0Pvvv69z5865HqXr9u3bWlxc1P7+vpn73osXL/T69Wv98ssv\nevDggYnFyM7OjiqViqrVqiqVionbKroSbnFxUc+fP9fu7u7Qz5/qlYRFvfzOhVar1b19KpWKWq2W\n44k6D6xff/21JKndbh/5ycilnZ2d7oO+FY1GQz///LM++eQT16NI6vzk8/HHH0uSVlZWuj/luhY/\nb1tbW3rvvffcDRMT/QTUarVM3Fabm5vdB/tz586NbOdUgebyu/HVarXuTx/b29vdO5drBwcHqtfr\n+uqrr1yP0tVut12PcEy73Vaz2VS9Xnc9iqRO/Pb29rSzs2Nmprhms6mlpSXXY0jqbNHOzc2pXC4f\n6ZVLs7Oz3V6+ePFCz58/H/r5U/8uDmtPDlq3vb2tjz76yMxq9cyZM1pZWdE333wz8sesPFhcPUtv\ntvj29vZMbE9J0tmzZ7urQWvPuTx69EhnzpxxPYakzkJyfn5ejUZDtVrNxPf5lStXulFutVo6e/bs\n0M+fKtAzMzPd1c6LFy80Ozs7zWGC0mw2devWLddjSOo8WET7cR9++KHW19cdT9T5Zt3Y2NDa2pra\n7baJ/cJGo9EN4OzsrIntqdnZWc3NzUnq3A+fPbP1suro+SkLGo2Grl27psXFRT148MDE9/nc3JyW\nl5e1s7OjmZmZkc+ZTRXo5eXl7jfroMvvXFhfX9fW1pa+/fZb16Mcsba21r1szMIqrNlsdh9g9/f3\nTTwBtri4qMXFRZVKJR0cHJh4srBSqejSpUuSOtt4Franrly50r3v7e/v68KFC44nesPCA1iv06c7\nL4SqVquamZlxPE3nJ8WtrS0tLCxof39fn3322dDPH/pClWEajUb3SS9+y91gm5ubWlpaUrlcVrvd\n1vr6uvMnnA4ODnT//n1JnTuVlZW9RdEKend3V9evX3c8TUej0VC5XNbW1papc7e7u6u7d+92n4C2\noF6vq1KpqN1um+lU9D117ty5kVueUwcaAJAtfmE/ABhFoAHAKAINAEYRaAAwikADgFEEGgCMItAA\nYNT/AwouH+ieTD9xAAAAAElFTkSuQmCC\n",
       "text": [
        "<matplotlib.figure.Figure at 0x4e49a50>"
       ]
      },
      {
       "metadata": {},
       "output_type": "display_data",
       "png": "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\n7yuLnarX62Ofv5F70LVaTdVq9cg10FYvv7t69aoqlYrrMY5oNpu6dOmSarWaWq2Wms2m65HUbDbV\nbDZVrVbVarX0448/uh6py9qWWaPR0Pvvv69z5865HqXr9u3bWlxc1P7+vpn73osXL/T69Wv98ssv\nevDggYnFyM7OjiqViqrVqiqVionbKroSbnFxUc+fP9fu7u7Qz5/qlYRFvfzOhVar1b19KpWKWq2W\n44k6D6xff/21JKndbh/5ycilnZ2d7oO+FY1GQz///LM++eQT16NI6vzk8/HHH0uSVlZWuj/luhY/\nb1tbW3rvvffcDRMT/QTUarVM3Fabm5vdB/tz586NbOdUgebyu/HVarXuTx/b29vdO5drBwcHqtfr\n+uqrr1yP0tVut12PcEy73Vaz2VS9Xnc9iqRO/Pb29rSzs2Nmprhms6mlpSXXY0jqbNHOzc2pXC4f\n6ZVLs7Oz3V6+ePFCz58/H/r5U/8uDmtPDlq3vb2tjz76yMxq9cyZM1pZWdE333wz8sesPFhcPUtv\ntvj29vZMbE9J0tmzZ7urQWvPuTx69EhnzpxxPYakzkJyfn5ejUZDtVrNxPf5lStXulFutVo6e/bs\n0M+fKtAzMzPd1c6LFy80Ozs7zWGC0mw2devWLddjSOo8WET7cR9++KHW19cdT9T5Zt3Y2NDa2pra\n7baJ/cJGo9EN4OzsrIntqdnZWc3NzUnq3A+fPbP1suro+SkLGo2Grl27psXFRT148MDE9/nc3JyW\nl5e1s7OjmZmZkc+ZTRXo5eXl7jfroMvvXFhfX9fW1pa+/fZb16Mcsba21r1szMIqrNlsdh9g9/f3\nTTwBtri4qMXFRZVKJR0cHJh4srBSqejSpUuSOtt4Franrly50r3v7e/v68KFC44nesPCA1iv06c7\nL4SqVquamZlxPE3nJ8WtrS0tLCxof39fn3322dDPH/pClWEajUb3SS9+y91gm5ubWlpaUrlcVrvd\n1vr6uvMnnA4ODnT//n1JnTuVlZW9RdEKend3V9evX3c8TUej0VC5XNbW1papc7e7u6u7d+92n4C2\noF6vq1KpqN1um+lU9D117ty5kVueUwcaAJAtfmE/ABhFoAHAKAINAEYRaAAwikADgFEEGgCMItAA\nYNT/AwouH+ieTD9xAAAAAElFTkSuQmCC\n",
       "text": [
        "<matplotlib.figure.Figure at 0x4e49a50>"
       ]
      }
     ],
     "prompt_number": 3
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Create Samples\n",
      "\n",
      "Using the `fipy_response` function, we can now create a sample set of microstructures and responses. We create `Nsample` microstrucures over a 2D space of $N \\times N$. We choose a very small system to first demonstrate the linear regression in real space."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "N = 10\n",
      "Nbin = 6\n",
      "Nsample = 5\n",
      "dt = 1e-3\n",
      "\n",
      "microstructures = np.array([np.random.random(N**2) for i in range(Nsample)])\n",
      "responses = np.array([fipy_response(m, dt=dt, N=N) for m in microstructures])\n",
      "print microstructures.shape\n",
      "print responses.shape"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "(5, 100)\n",
        "(5, 100)\n"
       ]
      }
     ],
     "prompt_number": 3
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Bin the Microstructure"
     ]
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "The function `bin`, discretizes the original microstructure $m^{\\prime}$ into a binned microstructure $m$, given by\n",
      "\n",
      "  $m^{\\prime}_{a, i} = m_{a, i}^h H^h$\n",
      "  \n",
      "where $H^h$ is the linear space for the discretization. The `bin` function takes $m^{\\prime}$ and returns $m$."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "def bin(m, N):\n",
      "    H = np.linspace(0, 1, N)\n",
      "    dh = H[1] - H[0]\n",
      "    return np.maximum(1 - abs(m[:,np.newaxis] - H) / dh, 0)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 4
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Run `bin` for each microstructure and rebuild the array (maybe this operation could be vectorized)."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "binnedMicrostructures = np.array([bin(m, Nbin) for m in microstructures])\n",
      "print binnedMicrostructures.shape"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "(5, 100, 6)\n"
       ]
      }
     ],
     "prompt_number": 5
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "The new `binnedMicrostructures` ($m_{a,i}^h$) has a shape of `(Nsample, N*N, Nbin)`. To double check that the binning worked we can evaluate $m_{a, i}^h H^h$ and check against the original $m^{\\prime}_{a,i}$. The summation is over the last axis (the binning axis)."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "H = np.linspace(0, 1, Nbin)\n",
      "reconstructedMicrostructure = np.sum(binnedMicrostructures * H[np.newaxis, np.newaxis, :], axis=-1)\n",
      "print np.allclose(reconstructedMicrostructure, microstructures)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "True\n"
       ]
      }
     ],
     "prompt_number": 6
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Real space\n",
      "\n",
      "In order to understand how to compute the influence coefficients it is useful to see a deomonstration in real space. Although we have a tensor representations of $m_{a, i}^h$, we need to create an intermediate matrix to calculate the dot product, $m_{a, i + j}^l \\alpha_j^h$. This matrix representation of $m$ is given by the `microstructureMatrix` and has shape `(N*N*Nsample, N*N*Nbin)`. The `microstructureMatrix` is essenially a [circulant matrix](http://en.wikipedia.org/wiki/Circulant_matrix) that isn't square. "
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "def rollMatrix(m, N, Nbin):\n",
      "    matrix = np.zeros((N**2, N**2 * Nbin))\n",
      "    for i in range(N**2):\n",
      "        matrix[i] = np.roll(m, -i, axis=0).swapaxes(0,1).flatten()\n",
      "    return matrix\n",
      "\n",
      "microstructureMatrix = np.concatenate([rollMatrix(m, N, Nbin) for m in binnedMicrostructures])\n",
      "\n",
      "print microstructureMatrix.shape\n"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "(500, 600)\n"
       ]
      }
     ],
     "prompt_number": 28
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "To calculate the influence coefficients, $\\alpha_j^h$, we use `numpy`'s `lstsq` function."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "responses = responses.flatten()\n",
      "coefficients = np.linalg.lstsq(microstructureMatrix, responses)[0]\n",
      "print coefficients.shape"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "(600,)\n"
       ]
      }
     ],
     "prompt_number": 9
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "If `Nbin > Nsample` then we can check that the influence coeffiencts exacly reproduce the `responses`. The result below should be `True` for an over-determined system."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "print np.allclose(np.dot(microstructureMatrix, coefficients), responses)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "True\n"
       ]
      }
     ],
     "prompt_number": 10
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "## Reciprocal Space"
     ]
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Having calculated the coefficients for a small system in real space,\n",
      "we will now calculate a much larger system them reciprocal space and\n",
      "then confirm that the produce a reasonable response. Of course, doing\n",
      "the calculation in reciprocal space drastically reduces the\n",
      "computational cost. The size of the least squares system is reduced\n",
      "from $ \\left(N^2 N_{\\text{sample}} \\times N^2 H \\right)$ to\n",
      "$\\left(N_{\\text{sample}} \\times H \\right)$ for each point in\n",
      "reciprocal space $N^2$. The convolution,\n",
      "\n",
      "  $ \\alpha_j^h m_{a,i + j}^h $\n",
      "  \n",
      "can be separated in Fourier space with\n",
      "\n",
      "  $ \\mathcal{F}_k \\left( \\alpha_j^h m_{a,i + j}^h \\right) =\n",
      "  \\mathcal{F}_k \\left( \\alpha_j^h \\right) \\mathcal{F}_k \\left(\n",
      "  m_{a,i}^h \\right) $\n",
      "  \n",
      " using the circular convolution theorem. If we write $P_{a,k}^h =\n",
      " \\mathcal{F}_k \\left( p^h_{a, i} \\right)$, $M_{a, k}^h = \\mathcal{F}_k\n",
      " \\left( m_{a,i}^h \\right)$ and $\\beta_k^h = \\mathcal{F}_k \\left(\n",
      " \\alpha_i^h \\right)$, then we just need to solve\n",
      " \n",
      " \n",
      "   $ P_{a,k}^h = \\beta_k^h M_{a, k}^h $\n",
      "   \n",
      "with a linear regression at each discretization location in $k$ space to\n",
      "calculate $\\alpha_i^h = \\mathcal{F}^{-1}_i \\left( \\beta_k^h \\right)$.\n",
      "\n",
      "In the example below we will create a microstructure with 101 bins and\n",
      "160 samples. This is enough to give reasonable influence coefficients."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "N = 20\n",
      "Nbin = 101\n",
      "Nsample = 160\n",
      "dt = 1e-3\n",
      "\n",
      "np.random.seed(101)\n",
      "microstructures = np.array([np.random.random(N**2) for i in range(Nsample)])\n",
      "responses = np.array([fipy_response(m, dt=dt, N=N) for m in microstructures])\n",
      "binnedMicrostructures = np.array([bin(m, Nbin) for m in microstructures])\n",
      "print microstructures.shape\n",
      "print responses.shape"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "(160, 400)\n",
        "(160, 400)\n"
       ]
      }
     ],
     "prompt_number": 10
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "We use `numpy`'s `fft2` to calculate the $P^h_{a,k}$ and $M^h_{k,a}$."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "microstructuresRS = binnedMicrostructures.reshape((Nsample, N, N, Nbin))\n",
      "responsesRS = responses.reshape((Nsample, N, N))\n",
      "fourierMicrostructures = np.fft.fft2(microstructuresRS, axes=(1, 2))\n",
      "fourierResponses = np.fft.fft2(responsesRS, axes=(1, 2))"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 11
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "fourierCoefficients = np.zeros((N, N, Nbin), dtype=np.complex)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 12
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "We calculate $\\beta_k^h$ at every point in $k$ space."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "for ii in range(N):\n",
      "    for jj in range(N):\n",
      "        fourierCoefficients[ii,jj,:] = np.linalg.lstsq(fourierMicrostructures[:,ii,jj,:], fourierResponses[:,ii,jj] )[0]\n",
      "        "
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 13
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "reconstructedResponses = np.sum(fourierMicrostructures * fourierCoefficients[np.newaxis], axis=-1)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 14
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "For a loose check let's see how close the reconstructed responses are to the sample responses."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "norm2 = np.linalg.norm(responsesRS.flatten() - np.fft.ifft2(reconstructedResponses, axes=(1, 2)).flatten())\n",
      "weight = np.linalg.norm(responsesRS)\n",
      "print norm2 / weight"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "0.000433339740246\n"
       ]
      }
     ],
     "prompt_number": 18
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "# Test data"
     ]
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "In the following, we will just check that one test response is reasonably close to a response calculated with the influence coefficients. We construct a `test_microstructre` and use the `fipy_response` function to calculate `test_response`."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "test_microstructure = np.random.random(N**2)\n",
      "test_response = fipy_response(test_microstructure, dt=dt, N=N)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 19
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "binned_test_microstructure = bin(test_microstructure, Nbin).reshape((N, N, Nbin))\n",
      "Fm = np.fft.fft2(binned_test_microstructure, axes=(0, 1))\n",
      "Fr = np.sum(Fm * fourierCoefficients, axis=-1)"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 20
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "calc_response = np.fft.ifft2(Fr, axes=(0, 1)).real.flatten()"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [],
     "prompt_number": 21
    },
    {
     "cell_type": "markdown",
     "metadata": {},
     "source": [
      "Just from observation, the numbers seem reasonable."
     ]
    },
    {
     "cell_type": "code",
     "collapsed": false,
     "input": [
      "print test_response[:20]\n",
      "print calc_response[:20]"
     ],
     "language": "python",
     "metadata": {},
     "outputs": [
      {
       "output_type": "stream",
       "stream": "stdout",
       "text": [
        "[-382.30241853  -12.40197819  228.26069825   40.52511079   98.21331131\n",
        "  238.72255462  264.15845687  332.52613342 -148.14998025 -164.96793467\n",
        "  -96.60157082  166.51657819  258.88340607  -68.45068609   58.41990692\n",
        "  393.40424284 -311.70006987  -64.01902205  220.11172271  277.84239844]\n",
        "[-382.12972641  -12.2371999   228.45580364   40.270015     98.14259533\n",
        "  238.54206123  264.35079052  332.97621213 -148.15938023 -165.07804659\n",
        "  -96.73003952  166.69179993  259.04245926  -68.66806771   58.60462215\n",
        "  393.06436852 -311.75985175  -64.08311151  220.4476074   277.78236929]\n"
       ]
      }
     ],
     "prompt_number": 22
    }
   ],
   "metadata": {}
  }
 ]
}