from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.sampler import Sampler
from scannerpy.stdlib import NetDescriptor
import os
import sys

video_path = sys.argv[1]
caffe_model_path = sys.argv[2]

movie_name = os.path.splitext(os.path.basename(video_path))[0]

with Database() as db:
    descriptor = NetDescriptor.from_file(db, caffe_model_path)
    batch_size = 8

    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    first_minute_frames = frame.sample()
    caffe_frame = db.ops.CaffeInput(
        frame = first_minute_frames,
        net_descriptor = descriptor.as_proto(),
        batch_size = batch_size,
        device = DeviceType.GPU)
    features = db.ops.Caffe(
        caffe_frame = caffe_frame,
        net_descriptor = descriptor.as_proto(),
        batch_size = batch_size,
        batch = batch_size,
        device = DeviceType.GPU)
    bboxes = db.ops.YoloOutput(
        caffe_output = features,
        device = DeviceType.CPU)
    output_op = db.ops.Output(columns=[bboxes, features])

    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            first_minute_frames: db.sampler.range(0, 60 * 30),
            output_op: 'bboxes'
        }
    )
    bulk_job = BulkJob(output_op, [job])
    
    [bboxes_table] = db.run(bulk_job, force=True)

    # output_table.profiler().write_trace('hist.trace')

    frame_bboxes = db.ops.Input()
    out_frame = db.ops.DrawBox(frame = first_minute_frames, bboxes = frame_bboxes)
    output_op = db.ops.Output(columns=[out_frame])
    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            frame_bboxes: bboxes_table.column('bboxes'),
            first_minute_frames: db.sampler.range(0, 60 * 30),
            output_op: movie_name + '_bboxes'
        }
    )

    bulk_job = BulkJob(output_op, [job])

    [out_table] = db.run(bulk_job, force=True)
    print(out_table.column_names())

    out_table.column('frame').save_mp4(movie_name + '_YOLO_boxes')
    print('Successfully generated {:s}_YOLO_boxes.mp4'.format(movie_name))
