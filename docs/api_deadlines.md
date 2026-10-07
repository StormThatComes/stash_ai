# Request visibility and read budgets

Long Stash requests report their fixed operation name, elapsed time, read/write
kind and the current in-flight count every five seconds. Credentials, variables,
scene/tag names and request payloads are never part of the heartbeat. Completion,
failure and cancellation remove the request and stop its ticker.

`REQUEST_TIMEOUT = (5, 600)` now means a maximum five-second connect timeout and
a **600-second total read budget**. All read attempts, backoff, redirects,
response bodies and GraphQL decoding share that deadline. Scalar timeout values
also set the total read budget. This applies to ordinary GraphQL reads, exactly
allowlisted SQL reads and same-origin media GETs.

Socket inactivity timeouts alone allow a trickling response to run indefinitely.
Only proven reads therefore perform blocking calls behind a caller deadline.
Header and body reads track their own connection sockets and interrupt them on
expiry or cancellation. A pooled connection already owned by a newer read is
protected from an older read's cancellation. Late responses are discarded.
At most four read workers can remain active per process, including a resolver
that cannot be interrupted; capacity waiting shares the caller deadline. A
single bounded close queue handles buffered readers. Late calls cannot apply
results, retry or issue mutations. Source state remains intact.

Closing prefetch propagates cancellation through preparation and media reads,
then joins the producer to release its images. The last complete prediction
batch is saved first; interrupted work cannot publish incomplete evidence.
Programmatic HTTP_RETRIES accepts integer 0..10, and REQUEST_TIMEOUT requires
positive finite scalar or (connect, read) values, excluding booleans.

Writes stay on the owning caller thread with the configured socket timeout.
They receive heartbeats but **are never automatically retried**. Durable pending
publication/cleanup intent continues to govern recovery of ambiguous writes.
