# Franka

Franka uses 8-dimensional actions and states, with `cam_front` and
`cam_right_wrist` mapped to HDF5 `front_cam` and `wrist_cam` datasets.

Place each task's episodes under `paths.data.franka/<task>/episode_<index>.hdf5`.
Required arrays are `action`, `observations/qpos`, and, for policy training,
`observations/images/front_cam` and `observations/images/wrist_cam`.
The task identifiers in the supplied YAML files must match the task directories.

Set machine paths through `configs/paths.local.yaml` or `TCLA_PATHS_CONFIG`.
The default language encoder is MiniLM (384 dimensions); configure local weights
through `paths.language_model`. Use the same language embeddings for AE and policy.

```bash
bash examples/franka/ae_train_files/train_version3.sh 0
```

Set `paths.action_ae.franka` to the resulting AE run directory, then train a policy:

```bash
bash examples/franka/policy_train_files/run_latent_baku.sh 0
bash examples/franka/policy_train_files/run_raw_baku.sh 0
bash examples/franka/policy_train_files/run_latent_act.sh 0
bash examples/franka/policy_train_files/run_raw_act.sh 0
```

Franka AE sampling requires two complete action windows (64 frames with the
reference configuration). Raw policy training requires no AE checkpoint.
