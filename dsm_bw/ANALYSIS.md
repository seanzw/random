# Hopper H200 distributed shared memory: evidence and working model

## Conclusion

The best current explanation of H200 DSM bandwidth is a 32-byte network lane
with a non-work-conserving per-SM allocation.  NVIDIA's patent groups six SMs
onto an `SM2SM4` CPC path.  Four 32-B/cycle lanes shared equally by six clients
give

```text
4 lanes * 32 B / 6 SM = 21.333 B/SM-cycle.
```

That number independently matches one-way DSM load, ordinary store, TMA put,
and 16-SM TMA-put scaling.  It also explains why an idle neighbor does not
raise one SM's rate and why 16 simultaneous puts reach approximately
`16 * 21.333 = 341.33 B/cycle` without requiring a physically wider per-SM
interface or a separate 1-GHz fabric clock.

The most plausible read protocol uses a 128-B response-data quantum, carried
as four 32-B flits.  In symmetric reads, the same per-SM directional allocation
must also carry roughly one read-command flit for the request in the opposite
direction.  This predicts 17.07 B/cycle per direction versus the maintained
benchmark's 15.37 B/cycle.  Accounting for measured one-way utilization
reduces the prediction to 15.90 B/cycle, only 3.4% above measurement.

Writes behave differently because payload travels with a write request and the
patent explicitly permits acknowledgment coalescing.  Ordinary stores and TMA
puts therefore lose only 4--6% per direction under symmetric traffic, while
reads lose 22--23%.  Mixed traffic proves that write data and read responses
nevertheless share the same directional physical resource.

This model is more economical than a 1-GHz DSM clock, a universally
half-duplex SM port, a 64-B physical flit, or a material token bucket.  The
32-B lane and 128-B read-response quantum remain inferred rather than directly
observed encodings.

## Measurement contract

All measurements launch exactly one CUDA thread-block cluster, so traffic is
confined to one GPC scheduling unit.  A 200-KiB dynamic shared-memory allocation
forces each cluster CTA onto a distinct physical SM; every sample rejects
duplicate `%smid` mappings.  Cluster sizes are 2, 4, 8, or 16, not multiple
GPCs.

Each size-slope point touches 16--96 KiB of unique addresses once.  This avoids
same-address replay, possible MSHR merging, and the impossible interpretation
that one SM stores a 64-MiB DSM tile.  TMA divides a tile into at most 16-KiB
commands and uses one remote `mbarrier` completion for the complete tile, with
no synchronization between commands.  Ordinary stores use one cluster
completion rendezvous per complete tile.  Checksum traffic is after the end
timestamp.

Every sample records both `%clock64` cycles and `%globaltimer` nanoseconds.
Steady rates come from fitting

```text
elapsed = fitted zero-payload intercept + payload_bytes * slope.
```

The intercept is only an affine-fit parameter.  It combines pipeline
fill/drain, command arbitration, completion latency, and small nonlinearities;
it is not a separately identified hardware event.  Raw one-shot 64-KiB
throughput is used only for equal-shape topology comparisons.

## Core bandwidth evidence

The maintained benchmark reproduced the exploratory 16--96-KiB size slopes.
All cycle-domain rates agree with the older implementation within 1.2%; all
nanosecond-domain rates agree within 2.0%:

| operation | one-way B/cycle | two-way B/cycle per direction | one-way GB/s | two-way GB/s per direction | duplex loss |
|---|---:|---:|---:|---:|---:|
| remote `ld.shared::cluster` | 19.87 | 15.37 | 35.83 | 27.55 | 22.6% |
| ordinary remote store | 18.87 | 18.12 | 33.68 | 32.41 | 4.0% |
| TMA remote put | 21.25 | 20.29 | 37.90 | 36.34 | 4--5% |

The cycle/ns slope ratio stays near 1.77--1.79 cycles/ns, consistent with the
1,785-MHz maximum SM clock reported by the H200.  The patent describes a
programmable amount per SM “per clock,” but does not state a fabric frequency
or clock-domain crossing.  No result requires a 1-GHz DSM domain.

Source-SM local shared-memory pressure is a useful bottleneck control.  With
the remote reader held constant, 1--16 local pressure warps, using either
bank-conflict-free accesses or a 128-B lane stride that maps onto the same four
banks, leave the remote-load completion at 3,628--3,633 cycles and 32.0--32.5
raw GB/s.  The target SRAM ports and a simple LSU stall are
therefore not the observed duplex bottleneck.

## Directional sharing and asymmetric read/write overhead

The mixed-direction runs moved two equal unique-address payloads at once.
Rank 0 always read rank 1, so read-response data traveled rank 1 -> rank 0.  A store or TMA
put carried its payload either in that same direction or in the opposite
direction:

| simultaneous payloads | same-direction aggregate | opposite-direction aggregate |
|---|---:|---:|
| read + ordinary store | 21.38 B/cycle, 38.21 GB/s | 29.70 B/cycle, 52.98 GB/s |
| read + TMA put | 21.45 B/cycle, 38.29 GB/s | 29.26 B/cycle, 52.20 GB/s |

Putting write data and read-response data in one physical direction collapses
their sum to the single-direction 21-B/cycle ceiling.  Reversing the write
payload raises aggregate throughput by 37--39%.  Ordinary store and TMA agree
within approximately 1%, locating the shared limit downstream of the issuing
instruction.  Virtual channels can own separate queues while still
multiplexing this physical resource.

A model in which all requests and responses share one universal half-duplex
port would also penalize symmetric writes by about 25%; measurement instead
shows 94--96% duplex efficiency.  The missing write penalty is consistent with
the patent's coalesced ACK mechanism: the target counts acknowledgments owed to
each source and can return a count when a threshold is reached or the return
bus is idle.  Reads cannot eliminate the request in the direction opposite
their response data.

## Scaling and topology

The scaling runs fitted TMA-put size slopes with every active rank putting one
tile to its ring neighbor.  The table uses the maintained implementation:

| active SMs | aggregate B/cycle | aggregate GB/s | per-SM B/cycle | per-SM GB/s |
|---:|---:|---:|---:|---:|
| 2 | 40.59 | 72.86 | 20.29 | 36.43 |
| 4 | 82.02 | 145.06 | 20.50 | 36.27 |
| 8 | 164.30 | 292.00 | 20.54 | 36.50 |
| 16 | 338.89 | 606.00 | 21.18 | 37.87 |

The 16-SM result is 8.35x the two-SM cycle rate and 8.32x the nanosecond rate,
versus ideal 8x scaling.  It is within 0.72% of the structural 341.33-B/cycle
prediction; the older implementation measured 341.94 B/cycle.  There is no
shared GPC put-data saturation through 16 SMs.  More importantly, the two-SM
case remains near 21 B/cycle per SM even
though most CPC clients are idle: spare allocation is not visibly reclaimed.
That favors fixed TDM, a hard per-SM issue cap, or a shaper with no material
burst allowance over a work-conserving crossbar.

Fixed-64-KiB concurrent loads from the maintained implementation expose
topology that isolated pairs do not.  All 37 retained configurations agree
with the older implementation within 3.3%:

| topology | aggregate GB/s |
|---|---:|
| 2-SM ring | 51.2 |
| 4-SM ring | 102.4 |
| 8-SM ring | 204.8 |
| 16-SM ring | 344.9--372.4, depending on stride |
| eight packed adjacent pairs | 390.1 |

The maintained run favors strides 2 and 14 (372.4 and 368.2 GB/s) over the
344.9-GB/s trough.  Eight disjoint adjacent pairs are about 11% above the
median 16-SM ring.  This implies a locality/arbitration boundary associated
with the two-SM grouping, but not a TPC-local bypass: the patent's illustrated
route still traverses GPCARB and GXBAR.

Two deliberately discarded searches constrain how much topology should be
modeled.  All 240 isolated directed pairs lie within 31.51--32.00 raw GB/s, and
a 32-B address-offset scan over 4 KiB changes cycles by only 0.39%.  Address
hashing and route choice become visible mainly under concurrency.  Exact hash
recovery is not necessary for a first simulator; a deterministic packet- or
cache-line-level interleave that can reproduce stride camping is sufficient.

## Packet and TDM inference

Assume one lane carries 32 B and the six CPC clients receive equal fixed
shares.  One lane flit then costs an average of `6/4 = 1.5` SM cycles.

For a 128-B read-response data quantum:

```text
one-way:  4 response flits * 1.5 cycles = 6 cycles
          128 B / 6 = 21.333 B/cycle

two-way: (4 response flits + 1 opposite read-command flit) * 1.5 = 7.5 cycles
          128 B / 7.5 = 17.067 B/cycle
```

Measurement gives 19.874 and 15.372 B/cycle.  Applying measured one-way
utilization to the ideal two-way prediction gives

```text
17.067 * (19.874 / 21.333) = 15.900 B/cycle.
```

The remaining difference is 3.4%.  Expressed per 128 B, measured cost rises
from 6.441 to 8.327 cycles.  The 1.886-cycle increment is close to one shaped
32-B flit (1.5 cycles); arbitration, tracker/credit bubbles, and metadata can
account for the residual.

Alternative response quanta under the same one-command-flit assumption are
less consistent with the measured duplex loss:

| response data | data flits | predicted two-way rate | ideal loss |
|---:|---:|---:|---:|
| 64 B | 2 | 14.22 B/cycle | 33.3% |
| 128 B | 4 | 17.07 B/cycle | 20.0% |
| 256 B | 8 | 18.96 B/cycle | 11.1% |

A warp-wide `ld.shared::cluster.v4.b32` returns 512 useful bytes and can
naturally decompose into four 128-B shared-memory wavefronts.  This makes 128 B
a plausible coalescing/response quantum, not a 128-B physical wire.

Short writes fit a different packet composition.  The patent shows four write
data elements, a 16-bit byte enable, source/destination SM identity, CGA/CTA
identity, address/offset, phase, and barrier metadata.  If each data element is
32 bits, useful short-write payload is 16 B inside a plausible 32-B packet.
Long writes and TMA can send a command followed by multiple data packets, and
ACK aggregation amortizes reverse metadata.

## Burst result: useful negative evidence

The discarded burst suite tested cold TMA commands, command trains, and two
32-KiB bursts separated by a source-side gap.  It did not reveal a material
token bucket:

- the cold single-command fit is 21.14 B/cycle;
- combined 4-KiB/16-KiB command trains fit 21.20 B/cycle and 37.89 GB/s;
- 128-B commands reach only 2.35 B/cycle because the TMA front end costs about
  54.5 cycles per command;
- after the first burst drains, completion grows one-for-one with inserted
  gap; the different gap knees are explained by command-issue time while the
  first burst is already draining.

The completion wait compiles to `TRYWAIT`, `YIELD`, and retry, quantizing
observations in approximately 147-cycle steps.  A bucket of only a few KiB or
less could be hidden by that resolution.  This distinction is not important
to the present model: the observable interface behaves as a hard
approximately 21.3-B/cycle per-SM allocation.

## Patent mapping

The relevant document is [US12248788B2, Distributed shared memory](https://patents.google.com/patent/US12248788B2/en)
(application US20230289189A1, 17/691,690).  The patent supports the following
structure:

```text
source SM LSU/TMA
  -> TPCARB and uTLB (address translation and route hash)
  -> GPCARB ingress (request queue, tracking, VC selection)
  -> CPC SM2SM4 path: six SM clients, four aggregate connections
  -> GX0/GX1 parallel 6x6 switch planes
  -> three CPC paths aggregated as SM2SM12
  -> destination GPCARB egress and destination SM
  -> read response or coalesced write ACK over a selected reverse route
```

GX0 and GX1 are parallel switch planes, not request and response VCs.  Firmware
may disable either plane and retain connectivity at half bandwidth.  The 12
GX-facing connections are aggregate network ports, not a one-to-one mapping to
the 16 CUDA cluster ranks.  The illustrated full GPC has three CPCs of six SM
slots each; H200 exposes 132 enabled SMs after product floorsweeping.

The patent defines blocking/request and nonblocking/response traffic, short
and long writes, multi-packet read responses, source outstanding-transaction
tracking, two phases around memory barriers, route hashing, bandwidth controls,
and coalesced ACK counts.  It does not specify physical flit width, exact
header encoding, link-level credit depth, arbitration schedule, fabric clock,
or an exact address hash.  Those details must not be presented as patent facts.

Virtual channels partition logical queues/buffers on multiplexed physical
links.  They prevent deadlock only with acyclic VC transitions or an escape
class.  Link credits count downstream buffer space and prevent overflow; they
do not provide fairness and cannot by themselves break cyclic dependencies.
The patent's endpoint “credit check” is outstanding transaction/completion
accounting and should not be conflated with unpublished per-hop NoC credits.

## Confidence boundary and simulator recommendation

| confidence | statement |
|---|---|
| measured | load duplex loss is 22--23%; store/TMA loss is 4--6% |
| measured | read responses and write payloads share a directional ceiling |
| measured | TMA puts scale linearly from 2 through 16 SMs at about 21 B/SM-cycle |
| measured | concurrent load topology has a favorable adjacent two-SM grouping and stride-dependent 16-SM contention |
| patent fact | six SM clients feed `SM2SM4`; three CPCs form `SM2SM12`; GX0/GX1 are parallel switch planes |
| strong inference | physical/logical lane unit is 32 B and each SM receives 2/3 lane |
| plausible inference | saturated reads return 128-B data quanta plus approximately one 32-B command flit |
| unresolved | literal periodic TDM versus an equivalent hard issue/shaping cap |
| intentionally deferred | exact route/address hash, physical metadata layout, per-hop credits, and any hidden fabric clock domain |

A useful first simulator should therefore implement:

1. three CPCs per full GPC, six SM clients per CPC, and four 32-B lanes per CPC;
2. a non-work-conserving 2/3-lane per-SM cap;
3. separate request/response virtual queues sharing directional lane service;
4. 128-B read responses plus request traffic, short and long writes, and
   coalesced write ACKs;
5. finite outstanding transactions and separate link-credit/backpressure and
   arbiter-fairness mechanisms;
6. a simple deterministic route interleave, with configurable stride camping.

That model captures every robust result without pretending the unresolved
hash or packet bitfields have been reverse engineered.

## Validation provenance

The retained evidence consists of source-SRAM pressure controls, independent
load/store/TMA size slopes, mixed-direction contention, 2/4/8/16-SM TMA-put
scaling, concurrent ring/packed topology, and negative isolated-pair,
address-offset, and burst searches.  After the code was made standalone, the
maintained smoke, bandwidth, and topology suites reproduced the relevant old
rates within 3.3%; the size slopes agree within 2.0%.
