I’d think of the front-end as a **fixed sensory transform** whose job is to turn “ink at pixels” into “stroke-like evidence,” so the learned silicon network does not have to rediscover locality and small-shift tolerance from scratch.

The key tradeoff is:

[
\text{more invariance} \Rightarrow \text{less spatial information}
]

For MNIST you want **small translation/slant/stroke-width tolerance**, not full invariance. A global Fourier magnitude, for example, throws away too much “where” information.

## Fourier/DCT: useful, but mostly as a coarse global descriptor

For a translated image,

[
g(x,y)=f(x-a,y-b)
]

the Fourier transform changes as

[
G(k_x,k_y)=e^{-i(k_x a+k_y b)}F(k_x,k_y)
]

So the **phase** carries the position shift, while the magnitude (|F|) is translation-invariant.

That means:

* keeping complex Fourier coefficients preserves location, but is shift-sensitive;
* using only (|F|) gives shift tolerance, but loses a lot of spatial layout;
* for digits, full Fourier magnitude can make different shapes too similar.

A better hardware-friendly version is the **2D DCT**, like JPEG:

[
I(x,y)\rightarrow C_{pq}
]

Keep only low-frequency coefficients, say (8\times 8) or (10\times 10). This gives a compact global shape vector. I would use DCT features as an **auxiliary channel**, not the whole representation.

A good input bundle might be:

[
\text{features}
===============

[
\text{pooled pixels},
\text{low-frequency DCT},
\text{local wavelet/edge energies}
]
]

## Wavelets are probably the best fit

Wavelets preserve locality while absorbing small shifts through pooling.

Conceptually:

[
x
\rightarrow x * \psi_{\theta,s}
\rightarrow |x * \psi_{\theta,s}|
\rightarrow \text{local average / pool}
]

where (\psi_{\theta,s}) is a fixed filter at orientation (\theta) and scale (s).

For your hardware, I would start with simple deterministic wavelets rather than full fancy Gabor filters:

### Cheap filters

Use filters like:

[
[-1, 0, +1]
]

horizontal and vertical gradients, plus diagonal variants.

Or Haar-like box filters:

[
\begin{bmatrix}
+1 & +1 & -1 & -1
\end{bmatrix}
]

[
\begin{bmatrix}
+1 & +1 \
-1 & -1
\end{bmatrix}
]

[
\begin{bmatrix}
+1 & -1 \
-1 & +1
\end{bmatrix}
]

These are easy with differential capacitor banks: positive rail for (+1), negative rail for (-1), then rectify.

For each filter response (z), represent unsigned edge energy as:

[
|z| \approx \operatorname{ReLU}(z)+\operatorname{ReLU}(-z)
]

That fits your positive/negative synapse architecture nicely.

## Local Fourier/DCT is better than global Fourier

Rather than one Fourier transform over the whole 28×28 image, split the image into coarse cells and compute a few local frequency components.

For example:

[
28\times 28
\rightarrow 7\times 7 \text{ cells}
]

For each cell or small neighborhood, keep:

* DC / local ink amount;
* horizontal contrast;
* vertical contrast;
* diagonal contrast;
* maybe one higher-frequency stroke feature.

This is basically a crude local DCT/wavelet transform. It preserves rough position, but makes the classifier less sensitive to exact pixel placement.

A very silicon-friendly version:

[
\text{cell features}
====================

[
\text{sum},
\text{left-right difference},
\text{top-bottom difference},
\text{diagonal difference}
]
]

For (7\times 7) cells:

[
7 \cdot 7 \cdot 4 = 196
]

features. That is a very reasonable front-end.

## Random filters are a strong option

Random fixed filters are especially attractive for your architecture because mismatch and device variation stop being purely bad.

Use many fixed random local receptive fields:

[
h_j=\operatorname{ReLU}(w_j^\top x-b_j)
]

where (w_j) is sparse and local, for example over a (4\times4), (5\times5), or (6\times6) patch.

Good choices:

* (w_{ji}\in{-1,0,+1})
* sparse connectivity, maybe 25–50% nonzero
* random threshold (b_j)
* local patch only, not full image
* train only the final readout layer

This turns your system into something like:

[
\text{fixed random retina/V1}
\rightarrow
\text{trainable classifier}
]

You do not need weight sharing. You just need enough random features spread across the image.

I’d combine deterministic and random filters:

[
h =
[
\text{pooled pixels},
\text{fixed Haar/Gabor-like filters},
\text{random local ReLU filters},
\text{low-frequency DCT}
]
]

Then train your capacitor readout weights.

## A concrete front-end I’d try

Start simple:

1. **Center / deskew / normalize ink**
2. **Blur and downsample**

[
28\times28 \rightarrow 14\times14
]

3. **Keep pooled pixels**

[
196 \text{ features}
]

4. **Compute local Haar features on a (7\times7) grid**

Maybe 4 filter types:

[
4 \cdot 7 \cdot 7 = 196
]

5. **Add low-frequency DCT**

Maybe first (8\times8), excluding or normalizing the DC term:

[
64 \text{ features}
]

6. **Optionally add random local filters**

Maybe 256–1024 random ReLU features.

So your classifier sees something like:

[
196 + 196 + 64 + 512 \approx 968
]

features, but many of those are fixed-front-end features, not trainable hidden synapses.

## My preferred ordering

If implementation cost matters, I’d rank them:

1. **Haar / local DCT / box-difference filters**
   Best simplicity-to-value ratio.

2. **Random sparse local ReLU filters**
   Very compatible with analog mismatch.

3. **Low-frequency global DCT**
   Useful extra global shape channel.

4. **Gabor / wavelet scattering-like filters**
   Stronger, but more circuit complexity.

5. **Global Fourier magnitude only**
   Interesting, but probably too invariant for digits.

The most promising silicon-friendly design is probably:

[
\boxed{
\text{deskewed image}
\rightarrow
\text{local fixed wavelet/Haar bank}
\rightarrow
\text{rectify}
\rightarrow
\text{pool}
\rightarrow
\text{random local features}
\rightarrow
\text{trainable readout}
}
]

That gives you much of what convolution gives you—local stroke detection and tolerance to small shifts—without requiring learned weight sharing.
