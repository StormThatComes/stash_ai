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
Streaming bodies are interrupted through the connection socket on expiry; late
responses are discarded/closed. A daemon network call can finish after the caller
has returned, but it cannot apply results, retry or issue mutations. The source
state remains intact and normal per-scene error handling controls publication.

Writes stay on the owning caller thread with the configured socket timeout.
They receive heartbeats but **are never automatically retried**. Durable pending
publication/cleanup intent continues to govern recovery of ambiguous writes.
