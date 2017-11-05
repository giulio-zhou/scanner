from scannerpy import Database, DeviceType, Job, BulkJob, ColumnType
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = sys.argv[1]

with Database() as db:
    batch_size = 1
    can_batch = batch_size > 1
    model_name = 'ssd_mobilenet_v1_coco'

    db.register_op('TfOp', [('input_frame', ColumnType.Video)],
                           ['boxes', 'scores', 'classes', 'num_detections'])
    print('yo')
    db.register_python_kernel(
        'TfOp', DeviceType.CPU, script_dir + '/kernels/tf_op.py',
        can_batch, batch_size)

    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    boxes, scores, classes, num_detections = db.ops.TfOp(
        input_frame = frame,
        batch_size = batch_size,
        batch = batch_size,
        model_name = model_name,
        device=DeviceType.CPU
    )
        
    output_op = db.ops.Output(columns=[boxes, scores, classes, num_detections])

    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            output_op: 'detections'
        }
    )
    bulk_job = BulkJob(output_op, [job])

    [output] = db.run(bulk_job, force=True, profiling=True)
