from scannerpy import Database, DeviceType, Job, BulkJob, ColumnType
from scannerpy.sampler import Sampler
import numpy as np
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = sys.argv[1]
frame_no_file = sys.argv[2]
output_dir = sys.argv[3]

with open(frame_no_file, 'r') as f:
    frame_nos = [int(s) for s in f.read().split(',')]

with Database() as db:
    load_video_from_scratch = False
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    sampled_frames = frame.sample()

    output_op = db.ops.Output(columns=[sampled_frames])
    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            sampled_frames: db.sampler.gather(frame_nos),
            output_op: 'selected_imgs'
        }
    )
    bulk_job = BulkJob(output_op, [job])

    [output] = db.run(bulk_job, force=True, profiling=True)
    output.profiler().write_trace('hist.trace')

    # Process outputs
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    video_name = os.path.splitext(video_path.split('/')[-1])[0]
    output_video_name = 'selected-' + video_name
    output.column('frame').save_mp4(output_dir + '/' + output_video_name)
