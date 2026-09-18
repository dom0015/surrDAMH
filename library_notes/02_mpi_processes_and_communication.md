# MPI processes and communication — notes

Review of the **working-tree** state (uncommitted changes included), read-only, no code executed.
Locations are `path:line`. Each finding is marked *confirmed by reading* or *suspected*.

## Files covered

Read fully: `surrDAMH/core.py`, `surrDAMH/configuration.py`, `surrDAMH/process_SAMPLER.py`,
`surrDAMH/process_SOLVER.py`, `surrDAMH/process_COLLECTOR.py`, `surrDAMH/process_CHILD.py`,
`surrDAMH/modules/communication.py`, `surrDAMH/modules/algorithm_interfaces_mpi.py`,
`surrDAMH/modules/algorithm_interfaces.py`, `surrDAMH/modules/algorithm_interfaces_local.py`,
`surrDAMH/modules/monitoring.py`, `surrDAMH/modules/tools.py`, `surrDAMH/modules/test_data.py`,
`surrDAMH/modules/continuation.py`, `surrDAMH/solvers.py`, `surrDAMH/solver_specification.py`,
`surrDAMH/stages.py`, `surrDAMH/modules/algorithms.py`.
Skimmed: `surrDAMH/surrogates/parent.py`, `surrDAMH/__init__.py`,
heads of `surrDAMH/distributions/independent_components.py` and `surrDAMH/post_processing.py`.
`out_*` directories were not opened.

## Process topology and roles

`Configuration.__post_init__` (`configuration.py:48-85`) derives the topology from
`MPI.COMM_WORLD.Get_size()` at construction time:

| use_collector | use_solvers_pool | no_samplers | rank_solvers_pool | rank_collector |
|---|---|---|---|---|
| True | True | size-2 | size-2 | size-1 |
| True | False | size-1 | None | size-1 |
| False | True | size-1 | size-1 | None |
| False | False | size | None | None |

* Sampler ranks are always `0 .. no_samplers-1` (`configuration.py:74`).
* Only assertion: `no_samplers > 0` (`configuration.py:73`) — the message says "use at least
  `mpirun -n 4`" but with defaults `-n 3` is accepted (one chain). Nothing validates
  `no_solvers`, `solver_maxprocs`, or that `no_samplers >= 1` is *useful*.
* `Configuration` must be built inside an MPI program; `__post_init__` is **not** re-run when the
  object is unpickled in a spawned child (`process_CHILD.py:27`), which is what makes the derived
  ranks correct there (the child's own COMM_WORLD is different). Corollary: the `paths_to_append`
  side effect (`configuration.py:52`, `87-90`) does **not** reach spawned children.
* Role dispatch: `core.py:118-145`. Every rank calls `run()`, then exactly one
  `COMM_WORLD.Barrier()` in `core.py:147`.
* `COMM_WORLD.Split` is called once per rank with color 0 (samplers, `process_SAMPLER.py:33`),
  1 (solver pool, `process_SOLVER.py:67`), 2 (collector, `process_COLLECTOR.py:102`). Only the
  sampler sub-communicator is used afterwards (per-stage `Barrier` and adaptive `Allreduce`,
  `process_SAMPLER.py:183,209`).
* Solver pool spawns children with `MPI.COMM_SELF.Spawn(sys.executable,
  args=[.../process_CHILD.py, solver_id, out_dir], maxprocs=conf.solver_maxprocs)`
  (`process_SOLVER.py:26-28`), once per `conf.no_solvers` → `no_solvers` **separate
  intercommunicators**, each with `solver_maxprocs` processes. Total OS processes =
  `size_world + no_solvers*solver_maxprocs`. In the child, all ranks run
  `set_parameters`/`get_observations`; only child rank 0 sends results back
  (`process_CHILD.py:60-64`).
* No-pool mode: each sampler builds its own `Solver` from `solver_spec` (`core.py:136-140`) into
  `output_dir/solver_output/rank{rank_world}`; the pool uses `.../rank{i}` for `i<no_solvers`
  (`process_SOLVER.py:71`) — same namespace, different meaning.

## Message protocol table

Payload "obj" = pickled Python object (lowercase mpi4py calls), "buf" = raw MPI buffer
(uppercase calls; datatype inferred from the numpy array).

### A. Sampler ↔ solvers pool (COMM_WORLD)

| # | sender → receiver | call / location | tag | payload | counterpart |
|---|---|---|---|---|---|
| A1 | sampler → pool | `Send(parameters)` `communication.py:351` | `1,2,3,…` (incrementing per evaluation) | buf, expected float64×`no_parameters` | `Iprobe` `process_SOLVER.py:102` + `Recv(received_data)` `:107` (float64×`no_parameters`) |
| A2 | sampler → pool | `Send(np.zeros((1,)))` `communication.py:370` | `0` = TAG_TERMINATE | buf, 1×float64 | same `Recv` `:107`; `tag==0` marks sampler inactive `:109-111` |
| A3 | pool → sampler | `send([obs, solver_tag])` `process_SOLVER.py:85` (only mode) | request tag | obj `[ndarray,int]` | `recv(source=pool)` `communication.py:401` (any tag) |
| ~~A4~~ | ~~pool → sampler~~ | ~~`Send(obs)` (raw mode)~~ | ~~`solver_tag`~~ | — | **REMOVED 2026-09-17** (decision 5 / WS8, with `Configuration.pickled_observations`). A3 is the only pool→sampler answer. |

Flow control: a sampler may have **one** outstanding request; `sampler_can_send` is cleared on
receipt (`process_SOLVER.py:108`) and set again when the answer is shipped (`:93`).

### B. Solver pool ↔ spawned child (per-child intercommunicator)

| # | direction | call / location | tag | payload | counterpart |
|---|---|---|---|---|---|
| B1 | pool → child | `bcast([conf, solver_spec], root=MPI.ROOT)` `process_SOLVER.py:32` | – | obj | `bcast(None, root=0)` `process_CHILD.py:27` |
| B2 | pool → child | `Bcast([tag,'i'], root=MPI.ROOT)` `:36` | – | buf int32 | `Bcast([tag,'i'], root=0)` `process_CHILD.py:44` |
| B3 | pool → child | `Bcast([data_par, MPI.DOUBLE], root=MPI.ROOT)` `:37` | – | buf float64×`no_parameters` | `Bcast([received_data], root=0)` `process_CHILD.py:50` |
| B4 | child(0) → pool | `send([sent_data, solver_tag], dest=0, tag=int(tag))` `process_CHILD.py:72` | request counter | obj | `Iprobe(tag=self.tag)` `process_SOLVER.py:45` + `recv(tag=self.tag)` `:39` |
| ~~B5~~ | ~~child(0) → pool~~ | ~~`Send(sent_data, dest=0, tag=solver_tag)` (raw mode)~~ | ~~`solver_tag`~~ | — | **REMOVED 2026-09-17** (decision 5 / WS8). B4 is the only child→pool answer. |
| B6 | pool → child | `Bcast(0)` + `Barrier()` + `Disconnect()` `process_SOLVER.py:59-61` | – | – | `process_CHILD.py:44-48` |

`tag==0` on B2 is the shutdown sentinel, so request tags start at 1 (`process_SOLVER.py:35`).

### C. Sampler ↔ collector (COMM_WORLD)

| # | direction | call / location | tag | payload | counterpart |
|---|---|---|---|---|---|
| C1 | sampler → collector | `Isend([idx])` `communication.py:47` | `3` TAG_UPDATE | buf 1×int64 | `Irecv` `communication.py:93`, re-posted `:105` |
| C2 | collector → sampler | `isend(evaluator)` `communication.py:128` | `4` TAG_EVALUATOR_OBJECT | obj (whole `Evaluator`) | `irecv(buf=max_buffer_size)` `communication.py:48`, waited `:60` |
| C3 | sampler → collector | `Send([idx])` `communication.py:70` | `5` TAG_STOP_UPDATING | buf 1×int64 | `Irecv` `communication.py:95`, polled `:114` |
| C4 | collector → sampler | `send(last_evaluator or None)` `communication.py:147` | `4` | obj | the sampler's still-posted `irecv`, waited in `:71` |
| C5 | sampler → collector | `isend([parameters, observations, weight])` `communication.py:179` | `10, 11, 12, …` per snapshot | obj (list) | `irecv(source, tag=current_idx)` `communication.py:224`, re-posted `:271` |
| C6 | sampler → collector | `Send([last_snapshot_idx])` `communication.py:197` | `0` TAG_TERMINATE | buf 1×int64 | `Irecv` `communication.py:221`, polled `:238` |

Unused constants: `TAG_READY_TO_RECEIVE=1`, `TAG_DATA=2` (`communication.py:16-17`).
Sampler↔pool tags and sampler↔collector tags live in the same communicator but never collide
because the peer ranks differ.

**Protocol invariant that makes C1–C4 safe** (verified by tracing): the collector polls
`sampler_requests_evaluator()` *only* when it has something new to send
(`process_COLLECTOR.py:231-234`), so "signals consumed" == "evaluators sent". Hence at shutdown
`current_idx == max_idx` exactly when the sampler's last posted `irecv` was already satisfied
(`communication.py:143-148`) — no missing counterpart. This invariant is implicit and undocumented;
moving the `sampler_requests_evaluator()` call out of the `if not sampler_got_last_evaluator[i]`
branch would deadlock every sampler.

## Lifecycle: start-up, main loops, termination

Start-up: all ranks construct `Configuration` (collective in practice: it reads COMM_WORLD size and
may read continuation files) → `SamplingFramework.run()` → `_configure_surrogate_gradients()` →
role dispatch → per-role `COMM_WORLD.Split`.

* **Sampler** (`process_SAMPLER.py:29-218`): builds `SolverMPI` (pool) or uses the local solver;
  if `use_collector`, builds `MpiEvaluatorProvider` with `request_initial_evaluator=True`
  (immediate C1) and `MpiSnapshotSink`; loops over stages; per stage picks proposal/algorithm and
  runs it; `comm_sampler.Barrier()` after each stage (`:209`). After the last stage that needs the
  surrogate: `commEvaluator.get_evaluator_and_terminate()` (C3) + `commSnapshot.terminate()`
  (waitall + C6) (`:197-204`). Finally `commSolver.terminate()` (A2), then
  `COMM_WORLD.Barrier()` twice (`:216-217`).
* **Solver pool** (`process_SOLVER.py:116-136`): `while any(sampler_is_active)`: poll eligible
  samplers with `Iprobe` (random permutation of `sampler_can_send`, `:96-97`), queue requests
  (`deque`, at most one entry per sampler), then for each child: if busy, `is_solved()` →
  forward result; if free, pop one request and `Bcast` it. Exits when every sampler has sent A2,
  then `terminate()`s all children (B6) and does two `COMM_WORLD.Barrier()`.
* **Collector** (`process_COLLECTOR.py:147-241`): `while any(needs_evaluator)`: drain snapshots
  (one per sampler per pass, until a pass yields nothing or `max_collected_snapshots_per_loop` is
  exceeded), out-of-sample quality check with the *previous* evaluator, `add_data`, retrain if
  `cond_init`/`cond_update`, then per sampler send a new evaluator on request and check for the
  stop signal. After the loop: write the two CSVs, `comm_s.terminate()` per sampler (blocks until
  C6 arrives and drains the remaining snapshots), two `COMM_WORLD.Barrier()`.
* **Barrier accounting**: every rank performs 2 role-internal barriers + 1 in `core.run:147` = 3,
  then 1 more in `write_report` (rank 0 at `core.py:310`, others at `:226`). Counts match, so the
  collective structure is correct **provided no rank raises**.

## Potential bugs and risks

| ID | Sev | Location | Description / why it matters | Confidence | How to verify |
|---|---|---|---|---|---|
| M1 | **high** | `core.py:152-165` | Dead method `temptemptemp` annotates parameters with `Iterable` and `Any`, which are **not imported** in `core.py` (imports: `core.py:4-23`) and there is no `from __future__ import annotations`. Annotations are evaluated at `def` time → `NameError` while importing `surrDAMH.core`, i.e. `import surrDAMH` fails on *every* rank. The body is copy-pasted from `post_processing.Samples.html_report_extended` (`post_processing.py:1621`, that module *does* import `Iterable, Any` at `:5`). | confirmed by reading | `python -c "import surrDAMH"` |
| M2 | **high** | `core.py:124` vs `modules/test_data.py:98` | `self.surrogate_test_data.as_surrogate_test_data(prior=..., likelihood=...)` — the method takes **no** arguments → `TypeError` on the collector rank, raised *before* `run_COLLECTOR` reaches `COMM_WORLD.Split` (`process_COLLECTOR.py:102`), so all other ranks hang in `Split`/`Barrier` instead of failing cleanly. Additionally a `TestData` built by `TestData.reuse()` has `log_posterior is None`, so the argument-less call would raise `ValueError` (`test_data.py:100`); the intended sequence is `compute_log_posterior_and_weights(prior, likelihood)` then `as_surrogate_test_data()`. | confirmed by reading | pass a `TestData` to `SamplingFramework(surrogate_test_data=…)` and run with a collector |
| M3 | **high** | `process_COLLECTOR.py:235` | `sampler_stops()` is polled only `if no_snapshots_used > 0`. If no surrogate is ever trained (`min_snapshots_initial` never reached, all stages with `send_snapshots_to_collector=False`, or an updater whose `train()` is never triggered), `needs_evaluator` never clears → collector spins forever while samplers sit in the final `COMM_WORLD.Barrier` → whole job hangs with no diagnostic. | confirmed by reading | run with `use_collector=True`, one MH stage, `min_snapshots_initial` larger than the number of snapshots the stage can produce |
| M4 | **high** | `algorithms.py:255-262` + `process_COLLECTOR.py:190` | A DAMH stage blocks in `get_evaluator()` until the first evaluator exists; the only snapshot source is the samplers themselves. If preceding stages do not deliver `min_snapshots_initial` snapshots, all samplers block and no further snapshots are produced — classic circular wait. Structural, not a coding slip, but there is no timeout, no warning, and the failure looks like a hung job. | confirmed by reading | `min_snapshots_initial` > snapshots from stage 0 |
| M5 | **high** | `modules/continuation.py:37`, `process_SAMPLER.py:65`, `communication.py:351` | `initial_sample_type="continued"` loads **float32** samples; `prior.transform` preserves dtype (`independent_components.py:42` `sample.copy()` + elementwise assignment; `identity` in `core.py:26` returns the same array). The first `Send` to the solver pool therefore transmits `MPI_FLOAT`×n (4n bytes) into the pool's `np.zeros(no_parameters)` float64 buffer (`process_SOLVER.py:81,107`) → type mismatch and silently wrong parameters for the initial evaluation (the chain state afterwards becomes float64 via the proposal). Continuation also loses precision by storing float32. | confirmed by reading | run a continued experiment with a solver that echoes its input; compare with the saved last sample |
| ~~M6~~ | med-high | ~~`process_CHILD.py:64`, `process_SOLVER.py:92`~~ | **FIXED 2026-09-17** (decision 5 / WS8) by deleting the raw path and `Configuration.pickled_observations`. Was: with `solver_returns_tag=True` **and** `pickled_observations=False`, a solver error produced `solver_tag < 0` used directly as an MPI tag → `MPI_ERR_TAG` / abort. The tag now always travels inside the pickled payload. Regression test: `tests/mpi/test_mpi_transport.py::test_i7_negative_solver_tag_completes_and_rejects` (was an xfail). | confirmed by test | — |
| M7 | med | `process_SAMPLER.py:12` | `from torch.mtia import snapshot` — stray, unused, and present in `HEAD` too. Makes `torch` a hard import dependency of every sampler rank and breaks on torch builds/versions without `torch.mtia`. | confirmed by reading | `grep -n mtia surrDAMH/process_SAMPLER.py`; import on a torch-free env |
| M8 | med | `process_COLLECTOR.py:126-132, 150, 186` | Preloaded `initial_snapshots` are counted twice when `updater.training_data_loaded` is False: `no_snapshots_total = N` (`:130`) and then `num_new_snapshots` starts at `N` (`:150`, reading the still-preloaded `list_new_snapshots`) and is added again (`:186`) → `2N`. Shifts `cond_init`/`cond_update` thresholds and corrupts the `snapshots_total` column in `surrogate_quality.csv`. The first quality row also mixes preloaded points into the "out-of-sample" batch. | confirmed by reading | pass `initial_snapshots` with N rows, print `no_snapshots_total` after the first loop |
| M9 | med | `process_SAMPLER.py:194-196` | `following_DAMH or following_onlySurr` is a Python `or` on two **lists**, not an elementwise or: whenever any stage remains, `following_onlySurr` is ignored. A later `use_only_surrogate` MH stage therefore does not keep the collector channel open. Currently masked because `commEvaluator.evaluator` still holds the last object (`:147`) and the collector side has already terminated cleanly — i.e. it works by accident. | confirmed by reading | stages `[DAMH, MH(use_only_surrogate=True)]` and check whether the surrogate stays frozen |
| M10 | med | `communication.py:30,48`; default `configuration.py:45` | `irecv(buf=self.max_buffer_size)` with `max_buffer_size = 1<<30` posts a **1 GiB** receive buffer per sampler, re-posted after every surrogate update. Also a hard ceiling: a pickled evaluator larger than this truncates (`MPI_ERR_TRUNCATE`). | suspected (mpi4py's int-as-bytecount semantics for `irecv(buf=…)` not verified from source here) | check RSS of a sampler rank right after start-up; or `MPI.COMM_WORLD.irecv(buf=1<<30, …)` in isolation |
| M11 | med | `process_SOLVER.py:116-128`, `process_COLLECTOR.py:147` | Both service loops are pure busy-wait (`Iprobe` / `Get_status`, no `sleep`, no blocking probe). Two cores are burned permanently; on oversubscribed nodes this directly steals time from the samplers and from the spawned solver processes. `process_SOLVER.py:99` shows the blocking alternative was disabled with `if False and …`. | confirmed by reading | `top` during a run; compare wall time with a 1 ms sleep inserted |
| M12 | med | `core.py:66-109` | `conf.use_surrogate_gradients` is silently mutated per rank, based on `surrogate_updater`/`surrogate_evaluator` which may be `None` on some ranks (a common pattern: build the updater only where it is needed). Result: samplers may keep gradients enabled while the collector disabled them (or vice versa) with no cross-rank consistency check; the warning is printed on rank 0 only. Changes proposal behaviour (Hamiltonian stages) → posterior-relevant. | confirmed by reading | construct `SamplingFramework` with the updater only on `rank_collector` and print `conf.use_surrogate_gradients` on each rank |
| M13 | med | `communication.py:69-77, 143-152` | The sampler's last `TAG_UPDATE` `Isend` is dropped without `Wait`/`Free` (`:73`), and the collector cancels the matching `Irecv` (`:144`) — leaving an outstanding request and possibly an unmatched message at `MPI_Finalize` (undefined per the standard; usually tolerated). `sampler_stops()` already cancels `request_update_signal` (`:115`) and `terminate()` cancels it a second time (`:144`). | confirmed by reading | run with `MPICH_..`/OpenMPI request-leak checking, or valgrind-mpi |
| M14 | med | `core.py:225-227` vs `:310`; `:142, 248, 293` | `write_report` is collective by construction (non-zero ranks block in `Barrier`). Any exception on rank 0 — `ValueError` at `:236`, missing sampling CSVs, a matplotlib failure — hangs the job instead of failing it. Separately, in pool mode `solver_instance` is `None` on rank 0 (set at `core.py:142`), so `par_names`, `field_statistics` and best-fit solver visualizations are silently skipped: the report content differs between pool and no-pool mode. Guards at `:248` and `:293` do handle `None` correctly (no crash). | confirmed by reading | run the pool-mode example and check the report for the field-statistics section |
| M15 | low-med | `communication.py:349, 171-187` | MPI tags grow without bound: one per full-model evaluation (`tag_solver`) and one per snapshot (`idx`). `MPI_TAG_UB` is only guaranteed to be ≥ 32767; long runs on a conservative MPI would fail mid-run. | confirmed by reading | `MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)` on the target machine; compare with expected evaluation count |
| M16 | low | `process_SOLVER.py:81, 107, 124` | `received_data` is a closure variable of `run_SOLVER` that the main loop **rebinds** to the popped queue entry (`:124`), so the `Recv` buffer identity changes between iterations. Currently safe (queue stores `.copy()`, `Bcast` is blocking), but a single change to non-blocking child sends would turn this into silent data corruption. | confirmed by reading | code inspection |
| M17 | low | `process_SOLVER.py:99` | `if False and all(child_can_solve):` — dead branch that was meant to block on `Probe` when no child is busy. Leftover debug code. | confirmed by reading | — |
| M18 | low | `configuration.py:48-52, 87-90` | `paths_to_append` is applied only in the process that *constructs* `Configuration`; spawned children unpickle `conf` without `__post_init__`, so `sys.path` is not extended there. A solver module whose own imports need those paths fails in the child (error text appears only in the spawned process's output). | confirmed by reading | spawn a solver whose module imports something that lives in `paths_to_append` |
| M19 | low | `configuration.py:73` | Assertion text ("use at least 'mpirun -n 4'") does not match the assertion (`no_samplers > 0`, i.e. `-n 3` with defaults). No validation of `no_solvers`/`solver_maxprocs` against available slots. | confirmed by reading | `mpiexec -n 3` with defaults |
| M20 | low | `process_COLLECTOR.py:141,144,244-267` | `surrogate_quality.csv` / `surrogate_quality_test.csv` are written only after the main loop; a killed or aborted run loses all surrogate-quality history, and the rows grow in memory meanwhile. | confirmed by reading | kill a run mid-way and look for the CSVs |
| M21 | low | `process_COLLECTOR.py:158` | `np.vstack` per received snapshot → O(n²) copying within a batch of up to `max_collected_snapshots_per_loop` (default 1000) and per-snapshot allocation churn. | confirmed by reading | profile the collector with a high snapshot rate |
| M22 | low | `communication.py:177, 128` | Every snapshot is `copy.deepcopy`'d and pickled individually (per accepted/rejected sample); every surrogate update pickles the **entire** `Evaluator` object once per sampler (for torch MLPs this includes the module and possibly device tensors). No batching, no parameter-only transfer. | confirmed by reading | time `send_evaluator` for the torch surrogate |
| M23 | low | `algorithms.py:150-162, 471` | `save_snapshots_to_file` rows are ragged (observations omitted when `None`) and heterogeneous: `"prerejected"` rows carry **surrogate** observations, and the `log_likelihood`/`log_prior` columns belong to the copied sub-chain state, not to a fresh exact evaluation. Consumers must filter by the first column. | confirmed by reading | inspect `sampling_output/raw_data/<stage>/rank####.csv` |
| M24 | note | `configuration.py:37` (working tree) | Default `state_dependent_approximation` changed `True` → `False` relative to `HEAD`. This changes the DAMH acceptance path (`algorithms.py:358-373`) and therefore acceptance rates / effective posterior. Flagged per project rules; not necessarily wrong, but it is an uncommitted silent default change. | confirmed by reading (`git diff -- surrDAMH/configuration.py`) | — |
| M25 | low-med | `algorithm_interfaces_local.py:207-208` + `algorithms.py:333` | `LocalEvaluatorProvider.evaluator_is_available()` returns `True` whenever *any* evaluator exists (not only when a new one is pending). In `use_collector=False` DAMH with `surrogate_model_updates=True`, `_refresh_surrogate_evaluator_if_needed` therefore reports "changed" on **every** sub-chain step, forcing the 3-evaluation branch of `_evaluate_surrogate_transition` instead of 1 → ~3× surrogate cost, no change in results. | confirmed by reading | count evaluator calls in local mode |

Also noted, not bugs: `run_COLLECTOR`'s `surrogate_delayed_init_data` parameter is never passed by
`core.run` (always `None`, `process_COLLECTOR.py:95-98`); `sampler_got_last_evaluator` switches
between `np.array` and `list` (`process_COLLECTOR.py:119,136,196`); `conf.debug` only gates two
collector prints.

## Suggested improvements

1. Delete `SamplingFramework.temptemptemp` (M1) — it has no callers and its docstring duplicates
   `post_processing.Samples.html_report_extended`.
2. Fix the `TestData` call (M2): `td.compute_log_posterior_and_weights(prior, likelihood)` then
   `td.as_surrogate_test_data()`; or give `as_surrogate_test_data` the optional
   `prior`/`likelihood` arguments the caller already assumes.
3. Make failure non-hanging: wrap each role body in `try/except` and call `MPI.COMM_WORLD.Abort(1)`
   on exception (at minimum on the collector and on rank 0 inside `write_report`). Today a single
   Python exception converts into a silent global hang (M2, M14).
4. Poll `sampler_stops()` unconditionally in the collector loop (M3); keep `no_snapshots_used > 0`
   only for the decision about *what* to send in `terminate()`.
5. Add a startup consistency check: broadcast `conf` (or a hash of the posterior-relevant fields,
   incl. `use_surrogate_gradients`, `state_dependent_approximation`, `transform_before_surrogate`)
   from rank 0 and assert equality on all ranks (M12).
6. Normalize dtypes at the MPI boundary: `np.ascontiguousarray(x, dtype=np.float64)` in
   `SolverMPI.set_parameters` and in `CommunicationWithChild.send_parameters` (M5), and store
   continuation samples in float64.
7. ~~Never use a solver status code as an MPI tag~~ — **done 2026-09-17** (decision 5 / WS8): the
   raw path is gone, the tag is always carried in the pickled payload (M6).
8. Replace the busy-wait loops with `Iprobe` + a small adaptive sleep, or with a blocking
   `Probe`/`Waitany` when nothing can progress (M11, and remove the `if False` remnant M17).
9. Lower the default `max_buffer_size`, or size it from a first, small "evaluator size" message
   (M10).
10. Collector: accumulate snapshots in Python lists and `vstack` once per batch (M21); flush the
    quality CSVs incrementally (M20).
11. Document the C1–C4 invariant ("the collector consumes an update signal only when it
    simultaneously sends an evaluator") as a comment in both `CommEvaluator_*` classes — it is the
    single thing preventing a shutdown deadlock.
12. Use `all()`/elementwise logic in `process_SAMPLER.py:194-196` (M9), e.g.
    `any(d or s for d, s in zip(following_DAMH, following_onlySurr))`.

## What should be tested or validated

No code was run for this review. Concrete checks to add:

* **Import smoke test** (catches M1/M7 immediately): `python -c "import surrDAMH"` in CI, plus
  `python -m pytest tests -q`.
* `mpiexec -n 3 / -n 4 / -n 8 python3 -m mpi4py toy_examples/minimal_example.py`
  (`use_collector=False`, `no_solvers=1`): expect clean exit, `no_samplers = n-1` chains, one
  spawned child, "Solver at spawned process - evaluations: k" printed once.
* `mpiexec -n 1` with `use_collector=False, use_solvers_pool=False` (`toy_examples/one_process_only.py`):
  single-process path must not touch any MPI point-to-point call.
* Collector path with `-n 4` and a DAMH stage after an MH stage: assert
  `sampling_output/surrogate_quality.csv` exists, its `snapshots_total` column is strictly
  increasing, and — with `initial_snapshots` of N rows — that its first value is N and not 2N (M8).
* **Deadlock regression tests with a wall-clock timeout** (e.g. `timeout 120 mpiexec …`, treat a
  timeout as a failure): (a) `min_snapshots_initial` larger than the snapshots stage 0 can produce
  (expect a clear error, currently a hang — M3/M4); (b) `use_collector=True` with a single MH
  stage and `send_snapshots_to_collector=False`.
* Small MPI unit tests for `communication.py` that can run on 2 ranks without the full framework:
  `CommSnapshot_sampler`/`CommSnapshot_collector` round-trip including the race where the
  TAG_TERMINATE message overtakes the last snapshot (the `<=` at `communication.py:248` claims to
  fix exactly this); `CommEvaluator_sampler`/`CommEvaluator_collector` shutdown in both branches of
  `terminate()` (`current_idx == max_idx` and `!=`).
* ~~Raw-buffer mode matrix: `pickled_observations ∈ {True, False}` × …~~ — obsolete since the raw
  path was removed (2026-09-17). What remains: `solver_returns_tag ∈ {True, False}` plus a solver
  that returns an error tag, which must now complete cleanly and reject the failed proposals
  (`tests/mpi/test_mpi_transport.py`).
* Continuation: run experiment A, continue as B with `initial_sample_type="continued"`, and assert
  that B's first evaluated parameter vector equals A's saved last sample bit-for-bit (M5).
* Print/assert `MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)` at start-up and compare with
  `max_evaluations`/expected snapshot count (M15).

## Open questions for the author

1. Is `temptemptemp` (`core.py:152`) meant to become a thin wrapper around
   `html_report_extended`, or can it be deleted?
2. What was the intended `TestData` API — should `as_surrogate_test_data` take `prior`/`likelihood`
   and compute the weights itself (`core.py:124`)?
3. Was `state_dependent_approximation`'s default flip (True → False) deliberate, and should it be
   committed with a note in the changelog?
4. Is `use_only_surrogate` supposed to keep receiving surrogate updates, or is the frozen-evaluator
   behaviour in `process_SAMPLER.py:147` intended?
5. ~~Is anyone still using `pickled_observations=False`?~~ **Answered (decision 5): no.** The raw
   path, the field and M6 were removed on 2026-09-17.
6. Should `write_report` stay collective (all ranks call it) or become rank-0-only with an explicit
   `comm.Barrier()` in the caller? The current contract is easy to break.
7. Is the pool intended to schedule more than one outstanding request per sampler? Today
   `sampler_can_send` caps it at one, so `no_solvers > no_samplers` cannot help.
8. Should `paths_to_append` be re-applied inside `process_CHILD` after unpickling `conf`?
