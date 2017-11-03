from scannerpy import Database, DeviceType, Job, BulkJob, ColumnType
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
video_path = sys.argv[1]

with Database() as db:
    db.register_op('PyInput', [('input_frame', ColumnType.Video)], ['frame'])
    db.register_python_kernel('PyInput', DeviceType.CPU,
                              script_dir + '/kernels/py_input.py')
    db.register_op('TfOp', [('input_frame', ColumnType.Video)], ['frame'])
    db.register_python_kernel('TfOp', DeviceType.CPU,
                              script_dir + '/kernels/tf_op.py')

    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    py_input = db.ops.PyInput(
        input_frame = frame,
        device=DeviceType.CPU
    )
    tf_out = db.ops.TfOp(
        input_frame = py_input,
        device=DeviceType.CPU
    )
        
    output_op = db.ops.Output(columns=[tf_out])

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
