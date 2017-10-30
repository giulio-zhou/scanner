from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.sampler import Sampler
from scannerpy.stdlib import NetDescriptor
import os
import sys
os.environ["GLOG_minloglevel"] = "1"

video_path = sys.argv[1]
caffe_model_path = sys.argv[2]

with Database() as db:
    descriptor = NetDescriptor.from_file(db, caffe_model_path)
    batch_size = 16

    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    caffe_frame = db.ops.CaffeInput(
        frame=frame,
        net_descriptor = descriptor.as_proto(),
        batch_size = batch_size,
        device=DeviceType.CPU)
    hashes = db.ops.Caffe(
        caffe_frame=caffe_frame,
        net_descriptor = descriptor.as_proto(),
        batch_size = batch_size,
        batch = batch_size,
        device=DeviceType.GPU)

    output_op = db.ops.Output(columns=[hashes])
    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            output_op: 'hashes'
        }
    )
    bulk_job = BulkJob(output_op, [job])
    
    [output_table] = db.run(bulk_job, force=True, profiling=True)
    output_table.profiler().write_trace('hist.trace')
