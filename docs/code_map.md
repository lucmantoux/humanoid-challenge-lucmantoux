# Code map

Labels are not in the filenames. `demo_001.mp4` … `demo_045.mp4` are listed in `data/raw/recording_log.csv` (`clip`, `type` = success / F1 / F2 / F3 / F4).

| Piece | Where |
|---|---|
| Load a video | `palm_prior.utils.video_frames`, called from `scripts/extract_human.observe` |
| MediaPipe hand | `palm_prior.perception.hand.detect` |
| Camera K | `data/calib/camera.yaml`, written by `scripts/calibrate.py` |
| ArUco pose | `palm_prior.perception.aruco.board_pose`, `static_pose` |
| Palm-size depth | `palm_prior.vision.hand_depth` |
| Lid / plate ray | `palm_prior.vision.objects.ray_plane` |
| QC grasp rule | `palm_prior.human.qc.grasp_passes` |
| Naive and two-anchor | `palm_prior.retarget.two_anchor`, `build_plan.build_plan` |
| Hybrid | `palm_prior.retarget.hybrid.hybrid_path` |
| Sim episodes | `palm_prior.sim.env.Env`, driven by `fix_all.stage_robot` |
| State | `palm_prior.state.build_state` (8) and `build_state_v2` (11) |
| World model | `palm_prior.wm.model.Ensemble`, `palm_prior.wm.train` |
| E1, E2 | `palm_prior.fix_all.stage_e1`, `stage_e2` |
| Config | `palm_prior.utils.load_config` reads `configs/default.yaml` |
