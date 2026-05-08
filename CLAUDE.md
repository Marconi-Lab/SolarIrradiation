# CLAUDE.md

## Code Style

### General Principles

- Write clean, readable code. Prioritize clarity over cleverness.
- Functions should be short (aim for <20 lines in the body). If a function
  does two things, split it into two functions.
- Names must be descriptive and intention-revealing. No abbreviations unless
  they are universally understood in the domain.
- No magic numbers or string literals. Extract them into named constants or
  config fields.
- Prefer explicit over implicit. Never rely on side effects for control flow.
- No commented-out code. Remove dead code; git history preserves it.

### Type Annotations

- Type-annotate every function signature, every return type, and every class
  attribute. No exceptions.
- Use `torch.Tensor` (not bare `Tensor`), `list[X]` (not `List[X]`),
  `dict[K, V]`, `tuple[X, ...]` with modern Python syntax.
- Use `Optional[X]` only when `None` is a semantically meaningful value, not
  as a default-avoidance pattern.
- Use `Protocol` or `ABC` for structural/nominal typing of interfaces.
- Generic classes should use `Generic[T]` with meaningful type variables.

### Classes and OOP

- All class members must be **private** (`_` prefix) or **protected** (`_`
  prefix) by default. Only expose through `@property` getters. Provide
  setters only when mutation is explicitly part of the design.
- Use `@property` (not methods) for structural attributes that describe what
  an object IS (e.g., `n_features`, `input_dim`, `params_type`). The `()`
  on a method call implies an action or non-trivial computation; a property
  signals an intrinsic characteristic.
- Use `@dataclass(frozen=True)` for all value objects and configuration.
  Mutable dataclasses need strong justification.
- Use **abstract base classes** (`ABC` + `@abstractmethod`) to define
  interfaces. Concrete implementations inherit from these.
- Use `Generic[T]` on ABCs when subclasses are parameterized by a type
  (e.g., a factory generic over its params dataclass). Expose the concrete
  type via a `@property` returning `type[T]` so that callers and tools can
  construct instances without knowing the concrete subclass.
- Use **polymorphism and inheritance** meaningfully. If two classes share
  behavior that varies in a specific dimension, define a base class with the
  shared logic and an abstract method for the varying part.
- Use **composition over inheritance** when the relationship is "has-a"
  rather than "is-a".
- Use `enum.Enum` (or `enum.StrEnum`) for categorical choices, never raw
  strings.
- Each class should have a single responsibility. If a class does two
  unrelated things, split it.
- Implement `__repr__` on all public-facing classes. For dataclasses this is
  automatic.

### Functions

- Functions should do one thing. If the function name contains "and", it
  probably does two things.
- Prefer pure functions (no side effects, deterministic output for given
  input) wherever possible.
- Limit function arguments to 5 or fewer. If more are needed, group them
  into a config or dataclass.
- Use keyword-only arguments (after `*`) for any argument that is not
  self-evident from position.

### Error Handling

- Raise specific exceptions with informative messages. Never use bare
  `except:` or `except Exception:` without re-raising.
- Validate inputs at public API boundaries. Internal functions can trust
  their callers.
- Use custom exception classes for domain-specific errors (inherit from
  appropriate built-in exceptions).

### Package Structure

- Organize code into clearly separated packages and sub-packages. Each
  package should represent a coherent domain concept.
- Every package must have an `__init__.py` that explicitly exports its public
  API via `__all__`.
- Avoid circular imports. If two modules need each other, extract the shared
  abstraction into a third module.
- Keep module files focused. A module with more than ~300 lines likely needs
  splitting.

---

## Architectural Design Principles

These principles govern how classes and modules are structured. They are not
optional style preferences; they are hard requirements that prevent the kind
of design drift that creates bugs and technical debt.

### Classes must mirror the domain, not the implementation

When designing a class, ask: "what is this thing mathematically / conceptually?"
The fields and methods should match that answer, nothing more.

- If a mathematical object is defined by X and Y, the class should contain X
  and Y. Do not add Z because some algorithm that consumes the object needs Z.
  That algorithm should derive Z from X and Y, or Z should live on the
  algorithm's side.

### No free-floating functions for logic that has state or identity

If a function constructs something, transforms something, or dispatches based
on type, it likely belongs as a classmethod, factory method, or method on an
enum. Free functions are acceptable only for pure stateless utilities (math
helpers, tensor manipulation).

- Factory functions that select a subclass based on an enum value should be a
  method on the enum itself, or a classmethod on the base class.
- If a function closes over parameters and returns a callable, make it a
  callable class with `__call__` so the captured parameters are inspectable
  via properties. Raw lambdas are acceptable only for throwaway one-liners in
  tests or examples, never in production code paths.

### Orthogonality: consumers must not know about producers

Modules should depend on interfaces, not on each other. Specifically:

- A downstream consumer (solver, encoder, renderer) should accept generic
  inputs (matrices, tensors, callables), not a domain object. The domain
  object can provide what the consumer needs via a method, but the consumer
  never imports the domain class.
- A neural network encoder should consume a tensor representation, not a
  symbolic domain object. The conversion between symbolic and tensor
  representations happens at an explicit boundary (a dedicated converter
  module), not inside the encoder.
- Notebooks and scripts should use the library's public API, not a
  reimplementation of internal logic. If the API is inconvenient, fix the
  API, do not duplicate it.

### Translate at boundaries; do not modify endpoints to fit each other

When two modules disagree on a convention (frame, units, schema), the right
fix is a translator class at the boundary, not patching either endpoint.
The endpoints stay correct relative to their own domain; the translator is
the only place that knows about both and reconciles them. This is sharper
than orthogonality: even when an endpoint *could* internalise a conversion
(e.g. via a flag), prefer an external translator class so the boundary is
named and isolated.

- Translators carry the assumption explicitly in their type/name (e.g. a
  `MetricToImperialAdapter` wrapping a metric-units sensor for an
  imperial-units consumer), so the cost of the translation is visible at
  the call site rather than hidden inside one of the endpoints.
- One endpoint with a configurable convention is a smell: the convention
  becomes a global setting that propagates through every call. Multiple
  thin translators are healthier.
- The "right" endpoint to leave alone is the one that is correct in its
  own domain — a third-party library staying bit-exact with its upstream,
  a data source faithfully representing the source format, etc. The
  translator is the only thing that needs to know both worlds.

### No silent fallbacks for missing data

If a loss function requires M >= 2 samples to compute variance, and it
receives M = 1, it must raise a ValueError, not silently return zero. Silent
fallbacks hide bugs. The caller made an error (passed insufficient data) and
must be told.

- Never return a zero tensor as a "safe default" for a loss that cannot be
  computed. This creates a gradient dead zone that is invisible during
  training.
- Never silently squeeze, unsqueeze, or reshape tensors to make shapes
  match. If the input has the wrong number of dimensions, raise a clear error
  stating the expected and actual shapes.
- When dispatching on a capability via `hasattr` or `Protocol`, the
  fallback branch (the object lacks the capability) must raise a clear
  error if the operation cannot produce correct results without the
  capability. A silent fallback that returns a plausible but incorrect
  result is worse than a crash. The only acceptable silent fallback is
  when the fallback behaviour is provably correct.

### Validation errors must contain the remediation

When a class rejects an input at construction or call time, the
`ValueError` (or other exception) message must tell the caller exactly how
to fix it. "X is not Y" is insufficient; "X is not Y. Wrap it in `Z(...)`."
is the minimum bar. This is especially load-bearing at numerically-critical
boundaries (frame conventions, unit conventions, schema mismatches), where
a silent wrong output is the alternative to a clear error.

- The reader of the error should not need to open the source file to fix
  the problem. Include the remediation snippet, the suggested adapter, or
  the exact configuration change.
- Pair with type-tagged contracts (enums, properties) so validation can
  catch mismatches at construction rather than after data has flowed.
- Do not auto-correct the situation silently — the user should see the wrap
  / convert / cast in their own code, not have it inserted under the hood.
  Auto-correction is exactly the silent-bug pattern this rule prevents.

### Parameters belong inside closures, not alongside them

When a function is parameterized (e.g., a scoring function with
coefficients), the parameters should be captured at construction time, not
passed separately at every call site.

- Bad: store `score_fn` and `score_params` as parallel fields, pass params
  to fn at every evaluation.
- Good: the scoring callable captures its params in a closure. The call
  site just invokes `score(state)` with no extra arguments.
- The closure should be a callable class (not a raw lambda) so that captured
  parameters are accessible via a `.params` property for serialization and
  debugging.

### String-keyed dicts are not a typed interface

When a function or factory accepts configuration through `dict[str, float]`
or `dict[str, Any]`, the caller gets no type checking, no IDE
autocompletion, and no safe refactoring. A misspelled key (`k_prodd` instead
of `k_prod`) silently produces a runtime KeyError, not a type error.

- If a function takes more than two related parameters that could be
  misspelled, group them into a frozen dataclass with named, typed fields.
- If a factory constructs objects from a parameter set, the parameter set
  must be a typed dataclass, not a dict. The factory should be generic over
  the params type.
- The single place where string-keyed dicts are converted to typed params is
  a dedicated `from_dict()` or `params_from_dict()` method. No other code
  should index into a params dict by string key.
- If two modules need to share parameter values, Python variable binding
  is sufficient when parameters are typed
  dataclass fields. No special "sharing" machinery is needed.

### Structural constants and identities belong inside the class

If a constant is intrinsic to a specific class, it must be a `ClassVar` on
that class, not a module-level variable. Module-level constants are acceptable
only for values that are genuinely module-wide (e.g., a logger instance, a
global registry).

- When multiple classes in the same module each have their own structural
  constants, module-level placement creates ambiguity about which constant
  belongs to which class.
- Constants that callers might want to customize should be exposed through
  the constructor with the `ClassVar` as the default, not frozen at module
  scope.

### Positional indexing into flat tensors is not an API

If a tensor has semantic slots (e.g., params[0] = rate_constant, params[1] =
k_m), the mapping must be defined in exactly one place (a frozen dataclass
with `to_tensor()` / `from_tensor()` methods). No code outside that dataclass
should index into the tensor by position.

### Notebooks must use the library, not reimplement it

If a notebook contains more than 10 lines of logic that duplicates a library
class (e.g., a training loop that mirrors the Trainer), that is a bug. The
notebook should call the library's public API. If the API does not support
what the notebook needs, extend the API.

- Trainer should return a result object that notebooks can plot directly.
- If a notebook needs a custom training variant, subclass the Trainer or
  pass configuration, do not copy-paste the loop.

### Distinguish structural properties from behavioral properties

A domain object often has two kinds of information: what it IS (structure)
and what it DOES (behavior). These should be represented separately, and
derived properties should not be conflated with defining properties.

- If two properties of an object seem related but can diverge in edge cases,
  they are separate concepts and must be stored or computed independently.
- Properties that can be computed from defining properties should be exposed
  as derived `@property` methods, not stored as redundant fields. If
  computation is expensive, use caching.

### Shared utilities live in neutral modules; layering must not invert

When two sibling packages need the same helper, accept the small refactor
to move it to a shared, neutral module rather than reaching across package
boundaries or duplicating the helper into each consumer. Cross-package
imports that invert the natural layering are a smell that compounds.

- A library package must never import from an application package. If a
  library needs something an application provides, that something is
  mis-located and belongs in a neutral top-level module that both can
  import from.
- Inlining a small helper into a second consumer to avoid the refactor is
  technical debt: the day the two copies diverge silently is the day the
  bug ships.
- "Neutral" means the module sits at or above the level of every consumer
  in the dependency graph. A helper used by two sibling packages belongs
  at the top of the namespace (e.g. `<project_root>/<helper>.py`), not
  inside one of them.

### Never import private symbols across module boundaries

If a symbol has a leading underscore, it is internal to its module. Other
modules must not import it. This is not a suggestion; it is a hard boundary.

- If module B needs to dispatch on the type of an object created by module A,
  use a protocol (`hasattr` check), a registry keyed on a public type
  (e.g., the params dataclass type), or a method on the object itself. Never
  use `isinstance` checks against private classes from another module.
- If you find yourself importing `_FooBar` from another module, stop. Either
  make it public (remove the underscore and commit to the interface) or
  redesign so the import is unnecessary.

### Prefer computable features over manual annotations for ML inputs

When deciding what information to feed into a neural network encoder, prefer
features that can be derived from the existing data structure over manually
annotated categories.

- Derived features (graph connectivity, statistical summaries, structural
  properties) are always consistent with the underlying data, cannot be
  mislabeled, and generalize to novel structures.
- Manual annotations require domain expertise, are incomplete by nature,
  and risk the model shortcutting structural learning in favor of memorizing
  labels.
- If the encoder struggles to distinguish important cases from structure
  alone, enrich the computable features (e.g., add edge features, add
  sensitivity estimates) before resorting to annotations.

### Structural constants must have a single source of truth

When a dimension, count, or schema is determined by one module and consumed
by another, define it in exactly one place and have all consumers derive
from that definition. Never duplicate a constant across files, even if both
files agree on the value today.

- If a feature vector has N channels, define an Enum listing the channels in
  the module that constructs the vector. Export `N = len(TheEnum)`. Consumers
  import N rather than hardcoding it.
- Never store structural constants (dimensions determined by data format) in
  config. Config is for hyperparameters (values the user chooses). A dimension
  dictated by the code is not a hyperparameter; it is a derived property.
- If adding a new channel to a feature vector requires changing more than two
  locations (the enum + the builder function), the abstraction is leaking.
- When a field has associated metadata (valid range, unit, description),
  co-locate the metadata with the field definition. For dataclass fields, use
  `dataclasses.field(metadata={...})`. A field name that appears in two
  places (once in the dataclass, once in a separate dict of metadata keyed by
  string) is a duplication bug waiting to happen.

### Network architecture details must be configurable, not hardcoded

The number of layers, hidden dimensions, activation functions, and
conditioning strategies in neural network components are hyperparameters.
They must live in config dataclasses, not in the constructor body.

- Never hardcode the number of hidden layers by writing out a fixed
  sequence of `nn.Linear` / activation pairs. Use a loop over a
  `n_hidden_layers` config value and store layers in `nn.ModuleList`.
- If a reusable pattern emerges (e.g., "MLP with conditioning at every
  layer"), extract it into its own `nn.Module` class with a clean
  constructor signature. Networks that share a pattern should be instances
  of that class, not copy-pasted `nn.Sequential` blocks.
- Conditioning (FiLM, concatenation, cross-attention) should be applied at
  every hidden layer of a conditioned network, not only at the output.
  Output-only conditioning limits the network to learning context-independent
  intermediate features, which is unnecessarily restrictive.

### Separate defining parameters from runtime inputs

A domain object's parameters (the values that define what it IS) must be
distinguished from the inputs it receives at execution time.

- A neural network is defined by its architecture and weights. The input
  tensor is a runtime argument to `forward()`, not a constructor parameter.
- A domain model is defined by its structure and parameters. Runtime inputs
  (initial states, time spans, solver settings) are passed separately at
  the call site, not stored in the same config or params object.
- When a factory constructs a domain object, its params dataclass should
  contain only defining parameters.
- This separation prevents conflation in sampling, serialization, and
  composition.

### Train sequential models on one-step objectives first

When training a model that generates sequences (trajectories, time series,
autoregressive outputs), prefer a one-step prediction loss with teacher
forcing as the primary training objective. Full-rollout losses (comparing
entire generated sequences against ground truth) should be used for
validation and optional fine-tuning, not as the primary training signal.

- Full rollout compounds early errors. By step T, the model operates on
  states far from the training distribution, producing noisy, uninformative
  gradients.
- Teacher forcing (always predicting from the true previous state) keeps
  every training step on-distribution and provides clean gradients.
- When a probabilistic model defines a tractable per-step likelihood (e.g.,
  Gaussian transitions in an Euler-Maruyama SDE), use the negative
  log-likelihood as the loss. This jointly trains mean and variance from a
  single principled objective, eliminating arbitrary loss-weighting
  hyperparameters.
- To bridge the gap between teacher-forced training and free-running
  inference, use scheduled sampling: start with 100% teacher forcing, then
  gradually increase the fraction of steps that use the model's own
  predictions.

---

## Documentation

- **Docstrings**: Google style on all public classes, methods, and functions.
  Include `Args:`, `Returns:`, and `Raises:` sections where applicable.
- Private/protected methods should have a brief one-line docstring explaining
  intent, unless the name is fully self-documenting.
- Module-level docstrings at the top of each `.py` file explaining the
  module's purpose in one to three sentences.
- No redundant docstrings that simply restate the function name (e.g.,
  `"""Gets the name."""` on a method called `get_name`).

---

## Testing

### Structure

Tests mirror the source tree exactly. Each source file
`src/<package>/foo/bar.py` has a corresponding `tests/foo/test_bar.py`.
Every test directory has an `__init__.py`.

### Philosophy

- **Test external interfaces, not internal behavior.** If a method is
  private (`_`-prefixed), test it through the public method that calls it.
- **Tests should explain why they exist.** Anyone reading a test should
  understand what contract it verifies. Use descriptive test names like
  `test_exponential_decay_matches_analytical`.
- **Use analytical references for numerical tests.** Prefer testing against
  known analytical solutions rather than cross-implementation comparisons.
- **Statistical tests use wide tolerances.** Stochastic convergence tests
  should use sufficient samples and wide tolerances to keep flaky failure
  rate below 1%.
- **Do not test trivial behavior.** Dataclass field defaults, enum string
  values, and `__repr__` methods do not need tests. Only test validation
  logic (e.g., `__post_init__` that raises on invalid inputs).

### Fixtures and Stubs

- Use `conftest.py` files to provide stub objects that satisfy interfaces
  without importing heavyweight modules.
- Downstream component tests should NOT depend on upstream components being
  correct — use fakes or stubs to isolate.
- Prefer factory fixtures over mutable shared state.

### Conventions

- Use `pytest`. No unittest-style classes unless grouping is genuinely
  needed.
- Use `pytest.fixture` for shared setup. Prefer factory fixtures over
  mutable shared state.
- Use `pytest.mark.parametrize` for testing the same logic across multiple
  inputs.
- Use `pytest.approx` or `torch.testing.assert_close` for numerical
  comparisons. Specify tolerances explicitly.
- Mark tests that depend on optional packages with appropriate
  `pytest.mark.skipif` guards.
- All tests must be fast (<5 seconds each). Use tiny configs and minimal
  data sizes.
- All tests run on CPU only.

### What to prioritize when writing new tests

1. **Data pipeline correctness** (corrupted inputs silently poison
   everything downstream).
2. **Loss function numerics** (subtle sign/normalization bugs are silent and
   devastating).
3. **Interface contracts** (type hierarchy, protocol compliance, batch vs.
   single-item consistency).
4. **Factory/builder correctness** (config-to-object mapping dispatches to
   the correct classes).

### What to skip

- `__repr__` methods.
- Dataclass field defaults (unless validation logic exists).
- Full training loop integration (too slow for unit tests).
- External service integrations (require live credentials).

---

## PyTorch Conventions

- All `nn.Module` subclasses must call `super().__init__()` first.
- Define sub-modules in `__init__`, compute in `forward`. No module creation
  inside `forward`.
- Name tensor dimensions in comments where shapes are non-obvious, e.g.,
  `# (batch, n_features, d_model)`.
- Use `torch.no_grad()` explicitly for inference and evaluation.
- Prefer `torch.nn.functional` for stateless operations; use `nn.Module`
  wrappers for stateful ones (layers with parameters).
- **Never loop over a batch dimension in Python.** If a computation applies
  the same operation to N independent items (N transitions, N samples, N
  trajectories), reshape them into a single batch tensor and run one forward
  pass. Python loops over individual vectors launch thousands of tiny GPU
  kernels; a single batched operation launches one. The speedup is typically
  10-100x. If a function needs to support both single-item and batched input,
  handle it via `dim()` checks at the entry point, not by looping internally.

---

## Git Practices

- Commit messages should be imperative and concise: "Add data loader"
  not "Added the data loader".
- One logical change per commit. Do not mix refactoring with feature
  additions.
- Run `tox` (or at minimum `pytest` + linting) before every commit.

