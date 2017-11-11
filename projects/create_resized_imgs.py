from scannerpy import Database, DeviceType, Job, BulkJob, ColumnType
from scannerpy.sampler import Sampler
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = sys.argv[1]
output_dir = sys.argv[2]

with Database() as db:
    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    sampled_frames = frame.sample()
    # Optional use a network to generate annotated images.
    if len(sys.argv) >= 4:
        model_name = sys.argv[3]
        batch_size = 1
        can_batch = batch_size > 1
        db.register_op('TfOp', [('input_frame', ColumnType.Video)],
                               [('frame', ColumnType.Video)])
        db.register_python_kernel(
            'TfOp', DeviceType.CPU, script_dir + '/kernels/tf_op.py',
            can_batch, batch_size)

        boxed_frame = db.ops.TfOp(
            input_frame = sampled_frames,
            batch_size = batch_size,
            batch = batch_size,
            model_name = model_name,
            device=DeviceType.CPU
        )
        resized_imgs = db.ops.Resize(
            frame = boxed_frame,
            width = 80,
            height = 80,
            device=DeviceType.GPU
        )
    else:
        resized_imgs = db.ops.Resize(
            frame = sampled_frames,
            width = 80,
            height = 80,
            device=DeviceType.GPU
        )

    output_op = db.ops.Output(columns=[resized_imgs])
    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            sampled_frames: db.sampler.gather([i for i in range(0, 1800, 1)]),
            output_op: 'resized_imgs'
        }
    )
    bulk_job = BulkJob(output_op, [job])

    [output] = db.run(bulk_job, force=True, profiling=True)
    output.profiler().write_trace('hist.trace')

    # Process outputs
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Process outputs into numpy array.
    downsampled_imgs = output.column('frame').load()
    # NOTE: Images appear to exit Scanner in BGR configuration.
    downsampled_imgs_npy = \
        np.array([img[1][:, :, ::-1] for img in downsampled_imgs])
    # Write numpy arrays to output directory.
    np.save('%s/data.npy' % output_dir, downsampled_imgs_npy)
