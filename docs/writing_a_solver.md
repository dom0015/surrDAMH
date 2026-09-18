# Writing a solver

A solver wraps the forward model `G`: parameters in, observations out. Subclass
`surrDAMH.solvers.Solver`.

## The `Solver` contract

```python
class MySolver(surrDAMH.solvers.Solver):
    def __init__(self, solver_id=0, output_dir=None, **your_params):
        ...
    def set_parameters(self, parameters):
        self.parameters = parameters
    def get_observations(self):
        ...  # run the model, return an (no_observations,) array
```

- `__init__(solver_id, output_dir, **kwargs)`: `solver_id` identifies this instance
  (e.g. the spawned child's rank, or the sampler rank running it locally) — key any
  scratch files by it if you write to disk, since the same class may be instantiated
  concurrently by several processes. `output_dir`, if given, is already created for you.
- `set_parameters(parameters)` / `get_observations() -> (no_observations,)`: the two-step
  contract; `__call__` and `set_parameters_and_get_observations` compose them. A solver
  instance is called repeatedly with different parameters — it must not depend on
  anything from a previous call beyond what `set_parameters` just set.
- `solver_tag`: if `Configuration.solver_returns_tag=True`, return
  `(observations, tag)` from `get_observations()` instead; `tag < 0` marks a failed
  solve (the solvers pool then substitutes zeros and forwards the negative tag).
- `visualize_solution(show=False) -> list[(figure, axes)]`: optional, called by
  `SamplingFramework.write_report()` on the best-fit sample; default returns `[]`.
- Optional duck-typed attributes (checked with `hasattr`, not required): `par_names`
  (parameter names, used in the HTML report), `field_builder`/`coords`/
  `measurement_points` (all three together enable the report's posterior-field-statistics
  section).

`parameters` received by `set_parameters` are already in PHYSICAL space (post
`prior.transform`, see `docs/concepts.md`) — a solver never sees internal-space samples.

## `SolverSpec`: constructing a solver out-of-process

```python
solver_spec = surrDAMH.solver_specification.SolverSpec(
    solver_module_path="/absolute/path/to/my_solver.py",
    solver_module_name="my_solver",
    solver_class_name="MySolver",
    solver_parameters={"some_kwarg": 1.0},
)
```

`get_solver_from_spec` imports `solver_module_path` fresh with
`importlib.util.spec_from_file_location` and instantiates
`solver_class_name(**solver_parameters, solver_id=..., output_dir=...)`. Required
whenever `Configuration.use_solvers_pool=True` (the pool rank loads the class once per
spawned child); optional when `use_solvers_pool=False` (an already-built
`solver_instance=` also works there).

**`solver_module_path` is stored absolute.** Since WS5 (2026-09-17), `SolverSpec`
resolves a relative path with `os.path.abspath` in `__post_init__`
(`SolverSpec.resolve_module_path()`, called again by `SamplingFramework.__init__` so that
subclasses defining their own `__init__` are covered too). The resolution happens on the
**launching rank, at construction time**, so what the solvers pool broadcasts to its
spawned children is always an absolute path. A relative path like
`"solver_examples/solver_examples.py"` (what the `toy_examples` specs use, hence the
documented `cd toy_examples/` step) is therefore resolved against the working directory
of the process that builds the spec — build the spec before any `os.chdir`.

What this does **not** fix: a spawned child does not inherit the parent's `sys.path`, so
if your *solver module itself* imports something that is only reachable via
`Configuration.paths_to_append`, the child still fails with a `ModuleNotFoundError`
(finding M18 — `paths_to_append` is applied only in the process that constructs
`Configuration`, and children unpickle `conf` without running `__post_init__`). Put such
directories on `PYTHONPATH` instead, which children do inherit.

## How the pool spawns children

With `use_solvers_pool=True`, the rank at `Configuration.rank_solvers_pool` spawns
`no_solvers` children via `MPI.COMM_SELF.Spawn(sys.executable, [process_CHILD.py,
solver_id, output_dir], maxprocs=solver_maxprocs)` and broadcasts `[conf, solver_spec]`
to each; every child then loops: receive parameters, call
`solver.set_parameters`/`get_observations`, send the result back, until it receives the
terminate signal. The pool rank itself never calls the solver directly — it only routes
requests from sampler ranks to whichever child is free. See `docs/running.md` for the
resulting process-count table and `library_notes/00_overview.md` §7 for the full
request/response protocol.
