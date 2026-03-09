basic_dir = /home/arshakumari/repos/XAI/src/Conv_NeXt/checkpoints.py.resolve()
tune_dir = basic_dir / "checkpoints" / run_id

tune_dir.mkdir(parents=True, exist_ok=True)

print("Saving checkpoints to:", tune_dir)