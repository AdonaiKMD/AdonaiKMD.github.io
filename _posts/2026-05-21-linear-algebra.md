---
layout: post
title: "Linear Algebra — The Chakra of Every Neural Network"
date: 2026-05-21
categories: [arc-I, foundations]
tags: [linear-algebra, PCA, math]
---

# Chapter 1 — Linear Algebra

### *From Zero to Hero: Building ChatGPT With My Own Two Hands*

> **Arc 1 — The Academy.** Every hero starts as a kid who can't do anything yet.
> Before Naruto could summon a single clone, he had to learn to mold chakra — the
> raw substance that *everything* later is built out of. Linear algebra is our
> chakra. It is the substance that neural networks, attention, embeddings, and
> eventually the whole of ChatGPT are made from. We're not going to memorize jutsu.
> We're going to learn to feel the chakra in our hands, so that later, when we build
> a transformer from scratch, it feels like breathing.
>
> Rank 0. No techniques learned yet. Let's begin.

---

## Table of Contents

1. [Why this chapter exists](#0-why-this-chapter-exists)
2. [The vocabulary of the dojo](#1-the-vocabulary-of-the-dojo)
3. [Linear systems and Gaussian elimination](#2-linear-systems-and-gaussian-elimination)
4. [Vectors and linear geometry](#3-vectors-and-linear-geometry)
5. [Vector spaces](#4-vector-spaces)
6. [Span, independence, basis, dimension](#5-span-independence-basis-dimension)
7. [Matrices](#6-matrices)
8. [Linear maps: the secret identity of every matrix](#7-linear-maps-the-secret-identity-of-every-matrix)
9. [Tensor operations](#8-tensor-operations)
10. [Determinants](#9-determinants)
11. [Eigenvectors, eigenvalues, and the spectral theorem](#10-eigenvectors-eigenvalues-and-the-spectral-theorem)
12. [Boss fight: deriving PCA from scratch](#11-boss-fight-deriving-pca-from-scratch)
13. [Exercise solutions](#12-exercise-solutions)

---

## 0. Why this chapter exists

A neural network is, at its core, a tower of two operations stacked over and over:
a **linear map** (multiply by a matrix) and a **nonlinearity** (squash the result).
Strip away the marketing and ChatGPT is billions of matrix multiplications wearing a
trench coat. If you do not deeply understand what a matrix *does* — not how to
multiply two of them, but what it *means* geometrically — then every later concept
(gradients, attention, embeddings) will feel like memorized spells instead of
understood technique.

So this chapter has one promise: by the end, when you see $W\mathbf{x}$ in a paper,
you will not see "weight matrix times input." You will see a geometric transformation
— a rotation, a stretch, a projection — acting on a point in space. That shift in
perception is the whole game.

**How to read this.** Every section has three layers:
- **Definition** — the precise statement, so you're never hand-wavy.
- **Intuition** — what it actually *means*, usually geometric.
- **Why it matters for ML** — the payoff, so nothing feels academic for its own sake.

Each section ends with exercises in three tiers: 🥋 **trivial** (warm-up, check you
read it), ⚔️ **genin** (real practice), and 🔥 **chunin** (these make you think).
Solutions are at the end — but a technique you copy is a technique you forget.

---

## 1. The vocabulary of the dojo

Before any technique, the words. You cannot learn what you cannot name.

| Term | Meaning |
|---|---|
| **Scalar** | A single number, no dimension. Written lowercase italic: $x$. |
| **Vector** | An object with both direction and magnitude. Lives in a vector space. |
| **Matrix** | A 2D array of numbers. Also — secretly — a linear map. |
| **Tensor** | A generalization: rank-0 = scalar, rank-1 = vector, rank-2 = matrix, and up. |
| **Tuple** | A finite *ordered* list where order matters and repeats are allowed: $(a_1, \dots, a_n)$. |
| **Function** | A relation sending each input to *exactly one* output. |

**Linear properties** — the two laws a map must obey to be called *linear*:

$$f(\mathbf{u} + \mathbf{v}) = f(\mathbf{u}) + f(\mathbf{v)} \quad\text{(additivity)}$$
$$f(\lambda \mathbf{u}) = \lambda f(\mathbf{u}) \quad\text{(homogeneity)}$$

Burn these two into memory. *Everything* in this chapter is downstream of them.

**Maps between sets** — the three words that describe how a function covers its target:

- **Injective (one-to-one):** distinct inputs go to distinct outputs. If $f(x_1) = f(x_2)$, then $x_1 = x_2$. The map never *merges* two inputs into one output.
- **Surjective (onto):** every element of the target is hit by at least one input. The image fills the whole codomain.
- **Bijective:** both injective and surjective — a perfect one-to-one correspondence. Bijective maps can be *undone* (they have inverses).

**Map terminology you'll meet again:**

- **Isomorphism:** a bijective map between two structures that *preserves the structure*. If two spaces are isomorphic, $V \cong W$, they are "essentially the same object wearing different clothes."
- **Endomorphism:** a map from a space *into itself*, $f: V \to V$. (A square matrix is one of these.)
- **Kernel** $\mathrm{Ker}(f)$: the set of inputs that get crushed to the zero vector. **Key fact:** $\mathrm{Ker}(f) = \{\mathbf{0}\}$ if and only if $f$ is injective.
- **Image** $\mathrm{Im}(f)$: the set of all possible outputs — everything the map can actually produce.

> 🥋 **Trivial.** Is $f(x) = 2x$ on the real numbers injective? Surjective?
> 🥋 **Trivial.** What is the kernel of the map $f(x) = 0$ (everything goes to zero)?
> ⚔️ **Genin.** A map $f: \mathbb{R}^3 \to \mathbb{R}^2$ can never be injective. Argue why in one sentence using the idea of dimension. (Intuition is fine; we prove it properly later.)

---

## 2. Linear systems and Gaussian elimination

> *The first jutsu: untangling many constraints at once.*

### 2.1 Linear combinations and linear equations

A **linear combination** of objects (numbers, vectors, functions) is a *weighted sum*:
each object multiplied by a scalar (its "weight") and added up. The weights decide
each object's relative importance — this single idea, "weighted sum," is the
heartbeat of every neuron you will ever build.

A **linear equation** in variables $x_1, \dots, x_n$ has the form

$$a_1 x_1 + a_2 x_2 + \cdots + a_n x_n = d$$

where $d \in \mathbb{R}$ is a constant and the $a_i$ are coefficients. A tuple
$(s_1, \dots, s_n)$ is a **solution** if substituting it makes the statement true.

> **Intuition.** Solutions of a *homogeneous* equation (where $d = 0$) always pass
> through the origin $(0, \dots, 0)$. That's exactly why they form vector subspaces
> — lines, planes — through the origin. Hold onto this; it returns constantly.

### 2.2 Gauss's method

**Theorem (Gaussian elimination).** If a linear system is transformed using any of
these three **elementary row operations**, the new system has the *same solution set*:

1. **Swap** two equations.
2. **Rescale** an equation by a nonzero constant.
3. **Row-combine:** replace an equation by itself plus a multiple of another.

The goal is to reach **echelon form**, where each row's **leading variable** (its
first nonzero coefficient) sits strictly to the right of the one above it, and
zero-rows sink to the bottom. Variables that are *not* leading are **free variables**
— they're the degrees of freedom in the solution.

**Reduced echelon form** goes further: every leading entry is $1$ and is the *only*
nonzero entry in its column. This form is unique, which is why it's the canonical
"answer."

### 2.3 General = Particular + Homogeneous

This is one of the most beautiful structural facts in all of linear algebra.

**Theorem.** The solution set of *any* linear system has the form

$$\{\, \mathbf{p} + c_1 \boldsymbol{\beta}_1 + \cdots + c_k \boldsymbol{\beta}_k \mid c_1, \dots, c_k \in \mathbb{R} \,\}$$

where $\mathbf{p}$ is *one* particular solution, the $\boldsymbol{\beta}_i$ form a
basis of the associated homogeneous system's solutions, and $k$ is the number of free
variables.

> **Intuition.** To describe *every* solution of a messy non-homogeneous system, you
> only need to find **one** solution, then add **every** solution of the clean
> homogeneous version. The particular solution shifts you to the right "location";
> the homogeneous part spans all the wiggle room. (This exact pattern reappears in
> differential equations and in the bias term of a linear layer.)

**Singular vs. non-singular.** A square coefficient matrix is **non-singular** if its
homogeneous system has *only* the zero solution (unique), and **singular** if it has
infinitely many. Non-singular = invertible = "loses no information." Keep this label;
it's the same idea as a nonzero determinant later.

> 🥋 **Trivial.** Put $\begin{cases} x + y = 3 \\ 2x + 2y = 6 \end{cases}$ into echelon form. How many solutions?
> ⚔️ **Genin.** Solve $\begin{cases} x + 2y - z = 1 \\ 2x + y + z = 5 \\ x - y + 2z = 4 \end{cases}$ by Gaussian elimination.
> ⚔️ **Genin.** For the plane $x + y + z = 0$, write the solution set in the "particular + homogeneous" form. (Hint: $\mathbf{p} = \mathbf{0}$ here — why?)
> 🔥 **Chunin.** A system with 3 equations and 4 unknowns is given. Without computing, what is the *minimum* number of free variables, and why does that guarantee infinitely many solutions (if any exist)?

---

## 3. Vectors and linear geometry

A **vector** is an element of a vector space — meaning we can add two of them and
scale them, obeying axioms we'll list next. Concretely, in $\mathbb{R}^2$ and
$\mathbb{R}^3$:

$$\mathbf{v} = \begin{pmatrix} x \\ y \end{pmatrix}, \qquad \mathbf{v} = \begin{pmatrix} x \\ y \\ z \end{pmatrix}$$

### 3.1 The fundamental operations

**A. Addition** (componentwise): $\mathbf{u} + \mathbf{v} = (u_1 + v_1,\ u_2 + v_2)^\top$.
Commutative, associative, with $\mathbf{0}$ as identity.

**B. Scalar multiplication:** $k\mathbf{v} = (kv_1,\ kv_2)^\top$. Distributes over
vector addition.

**C. Dot product (inner product):** the single most important operation in deep learning.

$$\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n = \sum_{i=1}^n u_i v_i$$

It does two jobs at once: it measures the **angle** between vectors, and it tests
**orthogonality** ($\mathbf{u} \cdot \mathbf{v} = 0$ means perpendicular). Geometrically,
$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\,\|\mathbf{v}\|\cos\theta$.

> **Why it matters for ML.** Every artificial neuron computes a dot product between
> its input and its weights, then squashes the result. "Attention" in a transformer
> is, at its heart, dot products between queries and keys measuring how *aligned*
> two tokens are. When you understand the dot product as "alignment," attention stops
> being mysterious.

**D. Norm (magnitude):** the length of a vector, $\|\mathbf{v}\| = \sqrt{\sum_i v_i^2}$.
This is the $L_2$ norm; it's the dot product of a vector with itself, square-rooted.

**E. Cross product** (3D only): $\mathbf{u} \times \mathbf{v}$ produces a vector
perpendicular to both. Less central to ML, but essential for graphics and physics
(and world models that reason about 3D).

**F. Transpose:** flips a column vector into a row vector and back. Trivial-looking,
but it's the bookkeeping that makes matrix shapes line up. Without the transpose,
you'd hit "shape mismatch" errors constantly — it's how we get dimensions to agree
in products.

> 🥋 **Trivial.** Compute $\begin{pmatrix}1\\2\end{pmatrix} \cdot \begin{pmatrix}3\\4\end{pmatrix}$.
> 🥋 **Trivial.** What is $\left\|\begin{pmatrix}3\\4\end{pmatrix}\right\|$?
> ⚔️ **Genin.** Find the angle between $(1,0)$ and $(1,1)$ using the dot-product formula.
> ⚔️ **Genin.** Show that scaling a vector by $k$ scales its norm by $|k|$.
> 🔥 **Chunin.** Prove the Cauchy–Schwarz inequality $|\mathbf{u}\cdot\mathbf{v}| \le \|\mathbf{u}\|\,\|\mathbf{v}\|$ in 2D, and state when equality holds. (Hint: consider $\|\mathbf{u} - t\mathbf{v}\|^2 \ge 0$ as a quadratic in $t$.)

---

## 4. Vector spaces

> *The dojo itself — the arena every technique lives inside.*

A **vector space** is a set $V$ with two operations (addition, scalar multiplication)
satisfying **ten axioms**. The point of the axioms is simply: *you can take linear
combinations and nothing breaks.*

**The ten axioms.** For all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and scalars $r, s$:

1. Closure under addition: $\mathbf{u} + \mathbf{v} \in V$
2. Commutativity: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
3. Associativity: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
4. Zero element: there exists $\mathbf{0} \in V$ with $\mathbf{v} + \mathbf{0} = \mathbf{v}$
5. Additive inverse: for each $\mathbf{v}$ there is $-\mathbf{v}$ with $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$
6. Closure under scaling: $r\mathbf{v} \in V$
7. Distributivity over scalars: $(r + s)\mathbf{v} = r\mathbf{v} + s\mathbf{v}$
8. Distributivity over vectors: $r(\mathbf{u} + \mathbf{v}) = r\mathbf{u} + r\mathbf{v}$
9. Associativity of scaling: $(rs)\mathbf{v} = r(s\mathbf{v})$
10. Scalar identity: $1 \cdot \mathbf{v} = \mathbf{v}$

> **Intuition.** Don't memorize all ten. Understand the *spirit*: a vector space is
> any place where addition and scaling behave "normally." Surprising things qualify —
> the set of all polynomials, the set of all $m\times n$ matrices, the set of
> continuous functions. This is why one theory covers all of them at once.

> 🥋 **Trivial.** Is the set of vectors in $\mathbb{R}^2$ with $x \ge 0$ a vector space? (Check closure under scaling by $-1$.)
> 🔥 **Chunin.** Show that the set of polynomials of degree *exactly* 2 is **not** a vector space, but degree *at most* 2 **is**.

---

## 5. Span, independence, basis, dimension

These four ideas are one idea seen from four angles: *how much space does a set of
vectors actually cover, and how efficiently?*

### 5.1 Subspace and span

A **subspace** is a subset of $V$ that is itself a vector space under the same
operations — a smaller dojo inside the big one.

The **span** of $S = \{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ is the set of *all* linear
combinations of those vectors:

$$\mathrm{span}(S) = \{ a_1 \mathbf{v}_1 + \cdots + a_k \mathbf{v}_k \mid a_i \in \mathbb{R} \}$$

It is the *smallest* subspace containing $S$ — everything you can reach by mixing
the vectors you have.

> **Constraints and degrees of freedom** (the intuition that makes this click).
> In $\mathbb{R}^3$ a point $(x,y,z)$ has **3 degrees of freedom**.
> - Impose **1 constraint** $x+y+z=0$: now $z = -x-y$ is forced. 2 free choices left → the solution set is a **plane** (dimension 2).
> - Impose **2 independent constraints**: 1 free choice left → a **line** (dimension 1).
>
> Every constraint you add removes one degree of freedom. This is *exactly* what's
> happening when a neural network layer projects high-dimensional data down — it's
> imposing constraints, collapsing degrees of freedom.

### 5.2 Linear independence

A set is **linearly independent** if no vector in it is a linear combination of the
others — there is *no redundancy*. Equivalently, the only way to write $\mathbf{0}$ as
a combination is with all-zero weights. If you *can* build one vector from the others,
the set is **dependent** — one of them is dead weight.

### 5.3 Orthogonal and orthonormal

- **Orthogonal:** $\mathbf{x}^\top \mathbf{y} = 0$ — perpendicular, at $90°$.
- **Orthonormal:** orthogonal *and* each has norm 1. The standard basis vectors
$(1,0,0), (0,1,0), (0,0,1)$ are the canonical example.

Orthonormal sets are the "nicest possible" coordinate systems — they show up in PCA,
in SVD, and in good weight initialization.

### 5.4 Basis and dimension

A **basis** is a sequence of vectors that is both:
1. **Linearly independent** (no redundancy), and
2. **Spanning** (covers the whole space).

A basis is a minimal, complete set of building blocks — the perfect amount: enough to
reach everywhere, with nothing wasted.

**Dimension** is the number of vectors in any basis. (Theorem: in a finite-dimensional
space, *all* bases have the same size — so dimension is well-defined.)

- $\dim(\mathbb{R}^n) = n$
- $\dim(P_n) = n+1$ (polynomials up to degree $n$; basis $\{1, x, x^2, \dots, x^n\}$)
- $\dim(M_{n\times m}) = n \cdot m$

### 5.5 Row space, column space, rank

For a matrix:
- **Row space:** the span of its rows. **Row rank:** that space's dimension.
- **Column space:** the span of its columns. **Column rank:** that space's dimension.

**Fundamental theorem:** row rank = column rank = **rank**. One number captures the
"true dimensionality" of a matrix's action.

> **Why it matters for ML.** Rank is *everywhere* in modern deep learning. **LoRA**
> (Low-Rank Adaptation), the dominant fine-tuning method, works precisely because the
> *update* to a weight matrix can be approximated by a low-rank matrix — far fewer
> numbers. Understanding rank is understanding why you can fine-tune a giant model on
> a single GPU.

> 🥋 **Trivial.** Are $(1,0)$ and $(2,0)$ linearly independent?
> ⚔️ **Genin.** Do $(1,1,0), (0,1,1), (1,0,-1)$ form a basis of $\mathbb{R}^3$? (Hint: check independence; one is a combination of the others.)
> ⚔️ **Genin.** What is the rank of $\begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$? What does that say about its invertibility?
> 🔥 **Chunin.** Prove that any set of $n+1$ vectors in $\mathbb{R}^n$ must be linearly dependent.

---

## 6. Matrices

An $m \times n$ **matrix** is a 2D array with $m$ rows and $n$ columns. Rows are
always counted first. Notation: bold capital, $\mathbf{X}$; entry at row $i$, column
$j$ is $X_{i,j}$; column $j$ is $\mathbf{X}_{:,j}$; row $i$ is $\mathbf{X}_{i,:}$.

### 6.1 Special matrices (and why they matter)

**Symmetric** ($\mathbf{X}^\top = \mathbf{X}$): square, mirrored across the diagonal,
so $X_{ij} = X_{ji}$.
> **Why ML cares:** symmetric matrices have *real* eigenvalues and show up as
> covariance matrices and graph Laplacians. The spectral theorem (Section 10) gives
> them a beautiful decomposition. PCA lives here.

**Identity** $\mathbf{I}_n$: ones on the diagonal, zeros elsewhere. Multiplying by it
changes nothing — the "do-nothing" map.

**Diagonal** $\mathbf{D} = \mathrm{diag}(x_1, \dots, x_n)$: nonzero only on the
diagonal. Multiplying by it just *scales each coordinate independently*:
$\mathrm{diag}(\mathbf{x})\mathbf{y} = \mathbf{x} \odot \mathbf{y}$.
> **Why ML cares:** per-feature scaling. This is the mechanism inside **batch
> normalization** and inside parts of attention.

**Orthogonal** $\mathbf{Q}$: columns form an orthonormal set. Equivalent statements
(all saying the same thing):
- $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$ (columns orthonormal)
- $\mathbf{Q}^{-1} = \mathbf{Q}^\top$ (the inverse is *free* — just transpose!)
- $\|\mathbf{Q}\mathbf{x}\| = \|\mathbf{x}\|$ (preserves lengths)
- $\det(\mathbf{Q}) = \pm 1$ ($+1$ rotation, $-1$ reflection)
> **Why ML cares:** orthogonal weight initialization keeps the norms of activations
> stable as signal flows through layers → no exploding/vanishing. And the $\mathbf{U},
> \mathbf{V}$ in SVD are orthogonal — they're pure rotations.

### 6.2 Matrix arithmetic

**Sum:** elementwise (same shape required), $C_{ij} = A_{ij} + B_{ij}$.

**Scalar multiplication:** $(\alpha A)_{ij} = \alpha A_{ij}$.

**Transpose:** $(A^\top)_{ij} = A_{ji}$. Key properties:
$(A^\top)^\top = A$, $(A+B)^\top = A^\top + B^\top$, and — *mind the order* —
$(AB)^\top = B^\top A^\top$.

**Frobenius norm:** the "size" of a matrix, $\|X\|_F = \sqrt{\sum_{i,j} X_{ij}^2}$ —
just the $L_2$ norm treating the matrix as one long vector.

### 6.3 Matrix multiplication

If $A \in \mathbb{R}^{m\times n}$ and $B \in \mathbb{R}^{n\times p}$, then
$C = AB \in \mathbb{R}^{m\times p}$. The **inner dimensions must match**, and each
entry is a dot product:

$$C_{ij} = \sum_{k=1}^n A_{ik} B_{kj}$$

> **Intuition.** $C_{ij}$ = (row $i$ of $A$) $\cdot$ (column $j$ of $B$). Matrix
> multiplication is just a grid of dot products. And as we'll see in Section 7, it's
> also *composition of transformations* — doing one map after another.

### 6.4 Matrix inversion

The inverse $A^{-1}$ *undoes* $A$: $AA^{-1} = A^{-1}A = I$.

**Invertible if and only if:** square, non-singular (independent columns), and
$\det(A) \neq 0$ — three phrasings of one condition.

> **Why ML cares.** In a linear model $\mathbf{y} = X\mathbf{w}$ we want the weights
> $\mathbf{w}$. If $X$ were square and invertible, $\mathbf{w} = X^{-1}\mathbf{y}$.
> In reality $X$ is rectangular (more data than features), so we use the **normal
> equation**:
> $$\hat{\mathbf{w}} = (X^\top X)^{-1} X^\top \mathbf{y}$$
> This is the closed-form solution to linear regression — your first real ML
> algorithm, and it's *pure linear algebra*. (In practice libraries avoid explicit
> inversion and use LU/QR/SVD decompositions, which are faster and numerically
> stable. Computing an inverse directly is almost always the wrong move in code.)

> 🥋 **Trivial.** Compute $\begin{pmatrix}1&2\\3&4\end{pmatrix}\begin{pmatrix}1\\1\end{pmatrix}$.
> 🥋 **Trivial.** What is $A^\top$ for $A = \begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}$? What shape is it?
> ⚔️ **Genin.** Verify $(AB)^\top = B^\top A^\top$ for two specific $2\times 2$ matrices of your choice.
> ⚔️ **Genin.** Find the inverse of $\begin{pmatrix}2&0\\0&3\end{pmatrix}$ without any algorithm — just by reasoning about what "undo" means for a diagonal matrix.
> 🔥 **Chunin.** Show that if $A$ and $B$ are both invertible, then $(AB)^{-1} = B^{-1}A^{-1}$. Why does the order flip?

---

## 7. Linear maps: the secret identity of every matrix

> *The reveal. This is the section that changes how you see everything.*

### 7.1 The definition

A map $h: V \to W$ is **linear** if it preserves both operations:

$$h(\mathbf{u} + \mathbf{v}) = h(\mathbf{u}) + h(\mathbf{v}), \qquad h(c\mathbf{v}) = c\,h(\mathbf{v})$$

> **Intuition.** Linear maps keep the origin fixed and send straight lines to straight
> lines. They can **stretch, rotate, or project** — but never bend or translate.
> Picture a grid: a linear map can shear it, spin it, squash it flat — but the grid
> lines stay straight and evenly spaced, and the origin doesn't move.

### 7.2 Every linear map IS a matrix (and vice versa)

Here is the punchline of the entire chapter.

**Recipe.** Pick a basis $B = \{\mathbf{e}_1, \dots, \mathbf{e}_n\}$ for the input
space. Then *any* linear map $h$ is represented by a matrix $H$ whose **columns are
the images of the basis vectors**:

$$H = \begin{pmatrix} | & | & & | \\ h(\mathbf{e}_1) & h(\mathbf{e}_2) & \cdots & h(\mathbf{e}_n) \\ | & | & & | \end{pmatrix}$$

**Why?** By linearity, for any $\mathbf{v} = \sum_j x_j \mathbf{e}_j$:

$$h(\mathbf{v}) = h\Big(\sum_j x_j \mathbf{e}_j\Big) = \sum_j x_j\, h(\mathbf{e}_j)$$

— which is *exactly* the matrix-vector product $H\mathbf{x}$. The matrix is nothing
but a lookup table of "where each basis vector lands." That's it. That's the whole
secret.

And the converse holds too: **any** matrix $A$ defines a linear map
$h(\mathbf{v}) = A\mathbf{v}$, automatically satisfying both linearity laws.

> **Why this is the most important idea in the chapter.**
> A linear layer in a neural network, $h(\mathbf{v}) = W\mathbf{v}$, is *literally*
> a linear map, and the weight matrix $W$ is its representation. When you train a
> network, you are **searching for the transformation** — the rotation/stretch/
> projection — that best reshapes your data so the next layer can separate it. You're
> not "adjusting numbers in a grid." You're *sculpting geometry*. Once you see weights
> as transformations, backpropagation becomes "nudge the transformation in the
> direction that reduces error," and the whole of deep learning snaps into focus.

### 7.3 Change of basis

A vector's coordinates aren't absolute — they depend on the basis you chose. Changing
basis is changing your *point of view* on the same underlying geometric object.

> **Why ML cares.**
> - **PCA** changes basis to align the axes with the directions of maximum variance —
>   same data, smarter coordinate system.
> - **Feature engineering** (raw pixels vs. Fourier coefficients, say) is choosing a
>   better basis for the task. Embeddings are learned bases.

> 🥋 **Trivial.** What matrix represents the identity map? The map that doubles every vector?
> ⚔️ **Genin.** Write the $2\times 2$ matrix that rotates the plane by $90°$ counterclockwise. (Hint: where do $(1,0)$ and $(0,1)$ land?)
> ⚔️ **Genin.** Write the matrix that projects every vector in $\mathbb{R}^2$ onto the $x$-axis. Is it invertible? Connect your answer to its kernel.
> 🔥 **Chunin.** A linear map sends $(1,0) \mapsto (2,1)$ and $(0,1) \mapsto (-1,3)$. Write its matrix, then compute where $(3,2)$ lands — *without* re-deriving anything, just by using the matrix.

---

## 8. Tensor operations

A **tensor** generalizes the ladder: rank 0 = scalar, rank 1 = vector, rank 2 =
matrix, rank 3+ = higher arrays (think: a batch of images, shape
[batch, channels, height, width]).

**Hadamard product** $\odot$ — elementwise multiplication of same-shape tensors:
$C_{ij} = A_{ij} \cdot B_{ij}$. In NumPy/PyTorch this is just `A * B`. (Contrast with
`A @ B`, true matrix multiplication — confusing these is the #1 beginner bug.)

**Reduction** — collapsing a tensor by summing (or max/min/mean) over axes. `X.sum(axis=0)`
sums down columns. Reductions are how you go from per-element values to a single loss
number.

**Dot product**, again — because it's *that* important:
$\mathbf{x} \cdot \mathbf{y} = \sum_i x_i y_i$. The most fundamental operation in deep
learning, computed inside every neuron.

> 🥋 **Trivial.** Compute $\begin{pmatrix}1&2\\3&4\end{pmatrix} \odot \begin{pmatrix}5&6\\7&8\end{pmatrix}$.
> ⚔️ **Genin.** Given a matrix $X$ of shape $(3,4)$, what shape results from summing over axis 0? Over axis 1?

---

## 9. Determinants

**Key idea.** The determinant of a square matrix $A$ is a single scalar that tells you
how the map $h(\mathbf{x}) = A\mathbf{x}$ scales **area** (2D) or **volume** (3D), and
whether it flips orientation.

For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

$$\det(A) = ad - bc$$

Geometrically this is the *signed area* of the parallelogram formed by the columns
$(a,c)$ and $(b,d)$.

- $\det(A) > 0$: orientation preserved.
- $\det(A) < 0$: orientation flipped (mirror).
- $\det(A) = 0$: the columns are collinear — the map **squashes space flat**,
  information is destroyed, and $A$ is **not invertible**.

**Essential properties:**
- Invertibility test: $\det(A) \neq 0 \iff A$ invertible.
- Volume scaling: $|\det(A)|$ is the volume-scaling factor.
- Multiplicative: $\det(AB) = \det(A)\det(B)$.
- Transpose: $\det(A^\top) = \det(A)$.
- Triangular/diagonal: the determinant is the *product of the diagonal*.

> **Computation tip.** To compute a large determinant, use row operations (the
> third kind leaves the determinant unchanged) to reach triangular form, then
> multiply the diagonal. This is how it's done in practice — never by the cofactor
> expansion you may have seen, which is exponentially slow.

> 🥋 **Trivial.** Compute $\det\begin{pmatrix}3&1\\2&4\end{pmatrix}$.
> 🥋 **Trivial.** What is $\det\begin{pmatrix}2&0&0\\0&5&0\\0&0&3\end{pmatrix}$? (Use the diagonal rule.)
> ⚔️ **Genin.** A matrix has $\det = 0$. What does that tell you about its columns, its rank, and its invertibility — all at once?
> 🔥 **Chunin.** Using $\det(AB) = \det(A)\det(B)$, prove that $\det(A^{-1}) = 1/\det(A)$ for an invertible $A$.

---

## 10. Eigenvectors, eigenvalues, and the spectral theorem

> *The final technique of the academy arc. Master this and you've understood the soul
> of a matrix.*

### 10.1 The core idea

Most vectors get knocked off their direction when you apply a matrix. But some special
vectors only get **stretched or shrunk** — they keep pointing the same way. Those are
the **eigenvectors**, and the stretch factor is the **eigenvalue**.

**Definition.** A nonzero vector $\mathbf{x}$ is an **eigenvector** of $A$ if

$$A\mathbf{x} = \lambda \mathbf{x}$$

for some scalar $\lambda$, the **eigenvalue**. The map doesn't rotate $\mathbf{x}$; it
only scales it.

**Reading $\lambda$:**
- $\lambda > 1$: stretch. $\quad 0 < \lambda < 1$: shrink.
- $\lambda < 0$: flip direction. $\quad \lambda = 1$: unchanged.
- $\lambda = 0$: $\mathbf{x}$ is crushed to $\mathbf{0}$ → the map isn't injective →
  $\det(A) = 0$. (Notice how every concept in this chapter keeps linking back to the
  others. That's not a coincidence — it's one structure seen from many sides.)

### 10.2 Eigenspace

For a given $\lambda$, the **eigenspace** $E_\lambda$ is all eigenvectors for that
$\lambda$ plus $\mathbf{0}$ — and it's exactly the kernel of $(A - \lambda I)$:

$$E_\lambda = \mathrm{Ker}(A - \lambda I) = \{\mathbf{x} \mid (A - \lambda I)\mathbf{x} = \mathbf{0}\}$$

### 10.3 How to compute them

1. **Characteristic polynomial:** $\chi_A(\lambda) = \det(A - \lambda I)$, a
   degree-$n$ polynomial.
2. **Eigenvalues:** solve $\chi_A(\lambda) = 0$. The roots are the eigenvalues.
3. **Eigenvectors:** for each $\lambda$, solve the homogeneous system
   $(A - \lambda I)\mathbf{x} = \mathbf{0}$.

### 10.4 Diagonalization

$A$ is **diagonalizable** if $A = PDP^{-1}$, where $P$'s columns are eigenvectors and
$D$ is diagonal with the eigenvalues.

> **Intuition for $A = PDP^{-1}$.** Applying $A$ becomes a three-step move:
> 1. $P^{-1}$: change into the eigenvector coordinate system.
> 2. $D$: just scale each axis (trivial — it's diagonal!).
> 3. $P$: change back to the original coordinates.
>
> A complicated transformation becomes "rotate into the right view, stretch, rotate
> back." This is the template for *every* matrix decomposition you'll meet.

**When possible?** $A$ is diagonalizable iff it has $n$ linearly independent
eigenvectors. Sufficient condition: $n$ distinct eigenvalues. Crucial special case:
**every real symmetric matrix is diagonalizable.**

### 10.5 The spectral theorem

**Theorem.** A real matrix factors as

$$A = Q \Lambda Q^\top$$

**if and only if** $A$ is symmetric. Here $Q$ is **orthogonal** (its columns are
*orthonormal* eigenvectors) and $\Lambda$ is diagonal with the real eigenvalues.

Because $Q$ is orthogonal, $Q^{-1} = Q^\top$ — the inverse is free. A symmetric matrix
acts as a pure stretch along perpendicular axes. Clean, beautiful, and the foundation
of the boss fight below.

> 🥋 **Trivial.** Verify that $(1,0)$ is an eigenvector of $\begin{pmatrix}3&0\\0&5\end{pmatrix}$. What's its eigenvalue?
> ⚔️ **Genin.** Find the eigenvalues of $\begin{pmatrix}2&1\\1&2\end{pmatrix}$ via the characteristic polynomial.
> ⚔️ **Genin.** For the same matrix, find an eigenvector for each eigenvalue. Confirm the two eigenvectors are orthogonal — and explain *why* the spectral theorem guaranteed that.
> 🔥 **Chunin.** Prove that the eigenvalues of a symmetric real matrix are real. (Hint: use $\mathbf{x}^* A \mathbf{x}$ with the complex conjugate and symmetry.)

---

## 11. Boss fight: deriving PCA from scratch

> *The first real jutsu, assembled from everything in the academy arc. We're going to
> build a genuine ML algorithm — Principal Component Analysis — using only the tools
> from this chapter. Nothing borrowed.*

**The problem.** You have data in high dimensions (say, every image is 784 numbers).
Most of that is redundant. You want to find a smaller set of axes that capture most of
the variation — to *compress* without losing much. This is PCA.

**The derivation, step by step, using only this chapter:**

1. **Center the data.** Subtract the mean so the cloud sits at the origin. (Linear maps
   fix the origin — Section 7 — so we must center first.)

2. **Build the covariance matrix** $C = \frac{1}{n} X^\top X$. It is **symmetric**
   (Section 6) — and that single fact unlocks everything.

3. **Apply the spectral theorem** (Section 10): because $C$ is symmetric,
   $C = Q \Lambda Q^\top$ with $Q$ orthogonal and $\Lambda$ diagonal, real eigenvalues.

4. **Interpret.** The columns of $Q$ (the orthonormal eigenvectors) are the
   **principal components** — the new perpendicular axes. The eigenvalues in $\Lambda$
   are the **variance captured along each axis** (Section 10's "$\lambda$ = strength
   of the stretch," now meaning strength of the spread).

5. **Compress.** Keep the eigenvectors with the largest eigenvalues; drop the rest.
   You've changed basis (Section 7) into the coordinate system where the data varies
   most, then thrown away the boring directions.

That's PCA. Every single ingredient — symmetric matrix, eigendecomposition, orthogonal
basis, change of basis, variance as eigenvalue — came from this one chapter. You didn't
memorize PCA; you *derived* it. That's the difference between a copied jutsu and one
you own.

> 🔥 **Chunin (capstone).** Implement PCA in ~15 lines of NumPy using only
> `np.cov`/`np.linalg.eigh` (note: `eigh`, the *symmetric* solver — now you know why!).
> Run it on any 2D blob of points you generate, and plot the principal axes over the
> data. This is your first blog-worthy artifact. Ship it.

---

## 12. Exercise solutions

> A technique you merely copy is a technique you forget. Try first; peek second.

**§1.** (1) Yes injective ($2x_1=2x_2 \Rightarrow x_1=x_2$), yes surjective (every real is $2\cdot(y/2)$), so bijective. (2) The whole real line — every input maps to 0, so $\mathrm{Ker} = \mathbb{R}$. (Genin) A map $\mathbb{R}^3 \to \mathbb{R}^2$ compresses 3 dimensions into 2, so by pigeonhole some distinct inputs must collide — it cannot be injective. (Made rigorous by the rank–nullity theorem.)

**§2.** (Trivial) The two equations are identical; echelon form has one leading variable and one free variable → infinitely many solutions (the line $x+y=3$). (Genin, 3×3) Eliminate to get $x=2, y=0, z=1$ (verify by substitution). (Genin, plane) $\mathbf{p}=\mathbf{0}$ works because $(0,0,0)$ satisfies $x+y+z=0$; the homogeneous part is the whole plane, spanned by e.g. $(1,-1,0)$ and $(1,0,-1)$. (Chunin) 4 unknowns − at most 3 leading variables ⇒ at least 1 free variable, so if any solution exists there are infinitely many.

**§3.** (Trivial) $1\cdot3 + 2\cdot4 = 11$. (Trivial) $\sqrt{9+16}=5$. (Genin) $\cos\theta = \frac{1}{\sqrt{2}} \Rightarrow \theta = 45°$. (Genin) $\|k\mathbf v\| = \sqrt{\sum (kv_i)^2} = |k|\sqrt{\sum v_i^2} = |k|\,\|\mathbf v\|$. (Chunin) $\|\mathbf u - t\mathbf v\|^2 = \|\mathbf u\|^2 - 2t(\mathbf u\cdot\mathbf v) + t^2\|\mathbf v\|^2 \ge 0$ for all $t$; a non-negative quadratic has discriminant $\le 0$, giving $(\mathbf u\cdot\mathbf v)^2 \le \|\mathbf u\|^2\|\mathbf v\|^2$. Equality iff $\mathbf u, \mathbf v$ are parallel.

**§4.** (Trivial) No — scaling $(1,0)$ by $-1$ gives $(-1,0)$ with $x<0$, outside the set. Not closed under scaling. (Chunin) Degree-exactly-2 fails closure: $(x^2) + (-x^2 + x) = x$, degree 1, leaves the set. Degree-at-most-2 is closed under both operations and contains $\mathbf 0$, so it's a space ($\dim = 3$).

**§5.** (Trivial) No — $(2,0) = 2\cdot(1,0)$, dependent. (Genin) Not a basis: $(1,0,-1) = (1,1,0)-(0,1,1)$, so they're dependent and span only a plane. (Genin) Rank 1 (second row = 2× first); rank < 2 ⇒ singular ⇒ not invertible. (Chunin) $n+1$ vectors in $\mathbb{R}^n$: arrange as columns of an $n\times(n+1)$ matrix; it has at most $n$ pivots, so at least one free column ⇒ a nontrivial dependence exists.

**§6.** (Trivial) $\binom{3}{7}$. (Trivial) $A^\top = \begin{pmatrix}1&4\\2&5\\3&6\end{pmatrix}$, shape $3\times 2$. (Genin) verify by direct computation. (Genin) $\begin{pmatrix}1/2&0\\0&1/3\end{pmatrix}$ — undo each independent scaling. (Chunin) $(AB)(B^{-1}A^{-1}) = A(BB^{-1})A^{-1} = AA^{-1} = I$; order flips because you must undo the *last* operation first (socks-then-shoes).

**§7.** (Trivial) $I$; and $2I$. (Genin) $\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ — $(1,0)\to(0,1)$, $(0,1)\to(-1,0)$. (Genin) $\begin{pmatrix}1&0\\0&0\end{pmatrix}$; not invertible — its kernel is the whole $y$-axis (nonzero vectors crushed to 0). (Chunin) $H = \begin{pmatrix}2&-1\\1&3\end{pmatrix}$; $H\binom{3}{2} = \binom{4}{9}$.

**§8.** (Trivial) $\begin{pmatrix}5&12\\21&32\end{pmatrix}$. (Genin) axis 0 → shape $(4,)$; axis 1 → shape $(3,)$.

**§9.** (Trivial) $3\cdot4 - 1\cdot2 = 10$. (Trivial) $2\cdot5\cdot3 = 30$. (Genin) Columns are linearly dependent (collinear), rank is deficient, matrix is not invertible — all three are the same statement. (Chunin) $\det(A)\det(A^{-1}) = \det(AA^{-1}) = \det(I) = 1 \Rightarrow \det(A^{-1}) = 1/\det(A)$.

**§10.** (Trivial) $\begin{pmatrix}3&0\\0&5\end{pmatrix}\binom{1}{0} = \binom{3}{0} = 3\binom{1}{0}$ ⇒ eigenvalue 3. (Genin) $\det\begin{pmatrix}2-\lambda&1\\1&2-\lambda\end{pmatrix} = (2-\lambda)^2 - 1 = 0 \Rightarrow \lambda = 1, 3$. (Genin) $\lambda=3$: $(1,1)$; $\lambda=1$: $(1,-1)$; their dot product is 0 — orthogonal, exactly as the spectral theorem promises for a symmetric matrix. (Chunin) For $A\mathbf x = \lambda\mathbf x$, take conjugate transpose: $\mathbf x^* A = \bar\lambda \mathbf x^*$ (using $A$ real symmetric). Then $\mathbf x^* A \mathbf x = \lambda \|\mathbf x\|^2$ and also $= \bar\lambda\|\mathbf x\|^2$, forcing $\lambda = \bar\lambda$, so $\lambda$ is real.

**§11.** See the 15-line implementation as your blog artifact; the key call is `np.linalg.eigh(C)` because $C$ is symmetric.

---

> **End of Chapter 1.** You walked in at Rank 0, unable to mold a single technique.
> You walk out having *derived PCA with your own hands* from nothing but axioms. The
> chakra is flowing. Next chapter: **Calculus** — how a network *learns*, which means
> understanding the derivative as the direction of steepest improvement. The gradient
> awaits.
>
> *Believe it.* 🍥
