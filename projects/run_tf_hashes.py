from scannerpy import Database, DeviceType, Job, BulkJob, ColumnType
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = sys.argv[1]
model_name = sys.argv[2]
output_dir = sys.argv[3]

with Database() as db:
    batch_size = 1
    can_batch = batch_size > 1

    db.register_op('TfOp', [('input_frame', ColumnType.Video)],
                           ['feature_vector'])
    db.register_python_kernel(
        'TfOp', DeviceType.CPU, script_dir + '/kernels/tf_op.py',
        can_batch, batch_size)

    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    sampled_frames = frame.sample()
    hashes = db.ops.TfOp(
        input_frame = sampled_frames,
        batch_size = batch_size,
        batch = batch_size,
        model_name = model_name,
        device=DeviceType.CPU
    )
    resized_imgs = db.ops.Resize(
        frame = sampled_frames,
        width = 120,
        height = 80,
        device=DeviceType.CPU
    )
        
    output_op = db.ops.Output(columns=[hashes, resized_imgs])

    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            sampled_frames: db.sampler.gather([i for i in range(0, 1800, 1)]),
            output_op: 'hashes'
        }
    )
    bulk_job = BulkJob(output_op, [job])

    [output] = db.run(bulk_job, force=True, profiling=True)
    output.profiler().write_trace('hist.trace')

    # Process outputs
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Process outputs into numpy array.
    feature_vecs = output.column('feature_vector').load()
    feature_vec_npy = np.array([np.loads(v[1]) for v in feature_vecs])
    print(feature_vec_npy, feature_vec_npy.shape)
    downsampled_imgs = output.column('frame').load()
    downsampled_imgs_npy = np.array([img[1] for img in downsampled_imgs])
    # Write numpy arrays to output directory.
    np.save('%s/feature_vectors.npy' % output_dir, feature_vec_npy)
    np.save('%s/data.npy' % output_dir, downsampled_imgs_npy)
