from scannerpy import Database, DeviceType, Job, BulkJob, ColumnType
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = sys.argv[1]
model_name = sys.argv[2]
output_dir = sys.argv[3]

with Database() as db:
    batch_size = 32
    can_batch = batch_size > 1

    db.register_op('CaffeOp', [('input_frame', ColumnType.Video)],
                              [('frame', ColumnType.Video)])
    db.register_python_kernel(
        'CaffeOp', DeviceType.CPU, script_dir + '/kernels/caffe_op.py',
        can_batch, batch_size)

    load_video_from_scratch = False
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    sampled_frames = frame.sample()
    video_frame = db.ops.CaffeOp(
        input_frame = sampled_frames,
        batch_size = batch_size,
        batch = batch_size,
        model_name = model_name,
        device=DeviceType.CPU
    )
        
    output_op = db.ops.Output(columns=[video_frame])

    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            sampled_frames: db.sampler.gather([i for i in range(0, 54000, 1)]),
            output_op: 'video'
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
