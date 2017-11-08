from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.sampler import Sampler
import numpy as np
import os
import sys
os.environ["GLOG_minloglevel"] = "1"

video_path = sys.argv[1]
output_dir = sys.argv[2]

with Database() as db:
    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    sampled_frames = frame.sample()
    resized_imgs = db.ops.Resize(
        frame = sampled_frames,
        width = 120,
        height = 80,
        device=DeviceType.GPU
    )

    output_op = db.ops.Output(columns=[resized_imgs])
    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            sampled_frames: db.sampler.strided(60),
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
    downsampled_imgs_npy = np.array([img[1] for img in downsampled_imgs])
    # Write numpy arrays to output directory.
    np.save('%s/data.npy' % output_dir, downsampled_imgs_npy)
