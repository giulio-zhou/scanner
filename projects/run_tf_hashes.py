from scannerpy import Database, DeviceType, Job, BulkJob, ColumnType
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = sys.argv[1]

with Database() as db:
    db.register_op('Test', [('frame', ColumnType.Video)], ['output'])
    db.register_python_kernel('Test', DeviceType.GPU,
                              script_dir + '/models.py')

    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    tf_frame = db.ops.Test(
        frame = frame,
        device=DeviceType.GPU
    )
    output_op = db.ops.Output(columns=[tf_frame])

    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            output_op: 'hashes'
        }
    )
    bulk_job = BulkJob(output_op, [job])

    [output] = db.run(bulk_job, force=True, profiling=True)
    # print(output.column_names())
    # print(output.num_rows())
