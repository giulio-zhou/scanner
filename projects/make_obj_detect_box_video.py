from scannerpy import Database, DeviceType, Job, BulkJob, ColumnType
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = sys.argv[1]
model_name = sys.argv[2]
output_dir = sys.argv[3]

with Database() as db:
    batch_size = 16
    can_batch = batch_size > 1

    db.register_op('TfOp', [('input_frame', ColumnType.Video)],
                           [('frame', ColumnType.Video)])
    db.register_python_kernel(
        'TfOp', DeviceType.CPU, script_dir + '/kernels/tf_op.py',
        can_batch, batch_size)

    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    sampled_frames = frame.sample()
    boxed_frame = db.ops.TfOp(
        input_frame = sampled_frames,
        batch_size = batch_size,
        batch = batch_size,
        model_name = model_name,
        device=DeviceType.CPU
    )
        
    output_op = db.ops.Output(columns=[boxed_frame])

    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            sampled_frames: db.sampler.gather([i for i in range(0, 1800, 1)]),
            output_op: 'detections'
        }
    )
    bulk_job = BulkJob(output_op, [job])

    [output] = db.run(bulk_job, force=True, profiling=True, pipeline_instances_per_node=1)

    # Process outputs
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    video_name = os.path.splitext(video_path.split('/')[-1])[0]
    output_video_name = model_name + '-' + video_name
    output.column('frame').save_mp4(output_dir + '/' + output_video_name)