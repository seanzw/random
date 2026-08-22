# H200 distributed shared-memory bandwidth

This standalone benchmark measures and reverse engineers Hopper SM-to-SM
distributed shared memory inside one thread-block cluster.  It is intentionally
separate from the generic TMA bandwidth examples in `../tma_bw`.

The main result, supporting evidence, patent mapping, and confidence boundary
are in [ANALYSIS.md](ANALYSIS.md).

## What remains in the benchmark

- 16--96-KiB size slopes for remote load, ordinary store, and TMA remote put,
  in one-way and symmetric two-way patterns;
- local source-SM shared-memory pressure while another SM performs remote
  loads;
- mixed read+store and read+TMA traffic in the same or opposite physical data
  direction;
- TMA-put size-slope scaling for 2, 4, 8, and 16 active SMs;
- concurrent load rings and packed 2/4/8-SM groups with stride sweeps.

Exploratory packet-step, address-offset, isolated 240-pair, barrier-subtraction,
repeated-epoch, and burst sweeps are not part of the maintained executable.
Their useful positive or negative conclusions are recorded in `ANALYSIS.md`.

## Files

- `benchmark.cu`: configurations, cluster launch, validation, and JSONL output
- `kernels.cuh`: remote load/store/TMA/mixed kernels and timer helpers
- `common.cuh`: minimal mbarrier helpers
- `process.py`: dependency-free median and size-slope analysis
- `ANALYSIS.md`: consolidated reverse-engineering result

Full historical raw outputs remain locally under `results/archive/legacy/` and
are ignored by Git.  Site-local scheduler scripts are also ignored because they
can contain account, partition, reservation, host, and filesystem details.

## Build and run

Compilation and GPU execution should happen inside an H200 allocation:

```bash
make
./dsm_h200.out --suite smoke --output smoke.jsonl
python3 process.py --input smoke.jsonl --output-dir smoke_analysis
```

Available suites are:

```text
smoke       one configuration for every maintained kernel/path
bandwidth   pair slopes, SRAM pressure, mixed traffic, and TMA scaling
topology    concurrent load ring/packed stride sweeps
all         bandwidth plus topology
```

Each result records `%clock64`, `%globaltimer`, physical SM IDs, per-rank
checksums, aggregate B/cycle, and aggregate GB/s.  `process.py` writes
`summary.csv`, `fits.csv`, and `topology.csv`.

## Standalone validation

The reorganized project was rebuilt and run independently on H200:

- all 8 smoke configurations passed their SM-placement and checksum checks;
- 4,710 bandwidth samples passed, with all retained cycle slopes within 1.2%
  and nanosecond slopes within 2.0% of the pre-refactor runs;
- all 37 topology configurations passed, with common aggregate rates within
  3.3% of the pre-refactor run.
