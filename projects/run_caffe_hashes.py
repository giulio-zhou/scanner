from scannerpy import Database, DeviceType, Job, BulkJob, ColumnType
import h5py
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

    db.register_op('CaffeOp', [('input_frame', ColumnType.Video)],
                              ['feature_vector'])
    db.register_python_kernel(
        'CaffeOp', DeviceType.CPU, script_dir + '/kernels/caffe_op.py',
        can_batch, batch_size)

    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    sampled_frames = frame.sample()
    hashes = db.ops.CaffeOp(
        input_frame = sampled_frames,
        batch_size = batch_size,
        batch = batch_size,
        model_name = model_name,
        device=DeviceType.CPU
    )
        
    output_op = db.ops.Output(columns=[hashes])

    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            # sampled_frames: db.sampler.gather([i for i in range(0, 20, 1)]),
            sampled_frames: db.sampler.all(),
            output_op: 'hashes'
        }
    )
    bulk_job = BulkJob(output_op, [job])

    [output] = db.run(bulk_job, force=True, profiling=True, pipeline_instances_per_node=1)
    output.profiler().write_trace('hist.trace')

    # Process outputs
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Process outputs into numpy array.
    size = output.column('feature_vector').size()
    dims = np.loads(next(output.column('feature_vector').load())[1]).shape
    hdf5_file = h5py.File('%s/feature_vectors.h5' % output_dir, mode='w')
    hdf5_file.create_dataset('data', [size] + list(dims), 'f4')
    for i, val in output.column('feature_vector').load():
        hdf5_file['data'][i] = val
    # feature_vecs = output.column('feature_vector').load()
    # feature_vec_npy = np.array([np.loads(v[1]) for v in feature_vecs])
    # print(feature_vec_npy, feature_vec_npy.shape)
    labels = np.arange(size)
    labels = labels.reshape(-1, 1)
    # Write numpy arrays to output directory.
    # np.save('%s/feature_vectors.npy' % output_dir, feature_vec_npy)
    np.save('%s/labels.npy' % output_dir, labels)
