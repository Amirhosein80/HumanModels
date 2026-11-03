# SMPL body models (not shipped)

HModel needs two SMPL model files in this directory, but they are **not
distributed with this repository** — the SMPL body models carry their own
license (registration required, non-commercial restrictions) which is
separate from this project's MIT license.

Download them from the official SMPL site (<https://smpl.is.tue.mpg.de/>),
then place them here as:

```text
models/SMPL_MALE.npz
models/SMPL_FEMALE.npz
```

They are ignored by git (`models/*.npz`). Without them, `SMPL(...)` and the
test suite cannot run:

```bash
uv run pytest tests/bmodels/   # fails until the models are present
```
