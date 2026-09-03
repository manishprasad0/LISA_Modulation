"""
Cluster GPU smoke test for simulator_gpu.py: generate a handful of samples
end-to-end (bbhx waveform generation on CUDA -> irfft -> STFT -> bumps array
-> saved .npz) to confirm the pipeline actually runs on the cluster's
cuda12x stack before committing to a real, much larger run.

Not a benchmark: batch_size here is just "small enough to trivially fit",
not a tuned value -- see simulator_gpu.generate_dataset's docstring for that.

Submitted via htcondor/test_run_gpu.sub; see that file and CLAUDE.md's
"MPI-IS cluster setup" section for the cluster environment this expects.
"""

from simulator_gpu import generate_dataset


N_SAMPLES = 5
OUTPUT_PATH = "outputs/test_run_gpu_dataset.npz"

if __name__ == "__main__":
    bumps, free_params = generate_dataset(
        n_samples=N_SAMPLES,
        output_path=OUTPUT_PATH,
        batch_size=N_SAMPLES,  # one batch, well within GPU memory at this size
        checkpoint_every=None,  # too few samples for periodic checkpointing to matter
    )
    print(f"saved {bumps.shape[0]} samples, bumps shape {bumps.shape} -> {OUTPUT_PATH}")
    for name, values in free_params.items():
        print(f"{name}: {values}")
