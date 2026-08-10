# Private regression samples

Place local benchmark inputs in this directory on the test server. Files in this folder are ignored by Git except for this README so device serial numbers and other sensitive document contents are not committed accidentally.

Current canonical filenames:

- `doc-text-001.pdf` — Turkish procurement text regression (`DOC-TEXT-001`)
- `device-label-001.jpg` — Philips device label regression (`DEVICE-LABEL-001`)

The Docker Compose stack mounts this directory read-only at `/data/regression-samples` inside `paddle-ui`.
