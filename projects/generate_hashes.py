from scannerpy import Database, DeviceType, Job, BulkJob
from scannerpy.sampler import Sampler
from scannerpy.stdlib import NetDescriptor
import os
import sys
os.environ["GLOG_minloglevel"] = "1"

video_path = sys.argv[1]
caffe_model_path = sys.argv[2]
output_dir = sys.argv[3]

with Database() as db:
    descriptor = NetDescriptor.from_file(db, caffe_model_path)
    batch_size = 16

    load_video_from_scratch = True
    if not db.has_table('target_video') or load_video_from_scratch:
        db.ingest_videos([('target_video', video_path)], force=True)

    frame = db.ops.FrameInput()
    sampled_frames = frame.sample()
    caffe_frame = db.ops.CaffeInput(
        frame=sampled_frames,
        net_descriptor = descriptor.as_proto(),
        batch_size = batch_size,
        device=DeviceType.CPU)
    hashes = db.ops.Caffe(
        caffe_frame=caffe_frame,
        net_descriptor = descriptor.as_proto(),
        batch_size = batch_size,
        batch = batch_size,
        device=DeviceType.GPU)
    resized_imgs = db.ops.Resize(
        frame = sampled_frames,
        width = 120,
        height = 80,
        device=DeviceType.CPU
    )

    output_op = db.ops.Output(columns=[hashes])
    job = Job(
        op_args={
            frame: db.table('target_video').column('frame'),
            sampled_frames: db.sampler.strided(10),
            output_op: 'hashes'
        }
    )
    bulk_job = BulkJob(output_op, [job])
    
    [output] = db.run(bulk_job, force=True, profiling=True)
    output.profiler().write_trace('hist.trace')

    # Process outputs
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Process outputs into numpy array.
    feature_vecs = output.column('hashes').load()
    feature_vec_npy = np.array([np.loads(v[1]) for v in feature_vecs])
    print(feature_vec_npy, feature_vec_npy.shape)
    downsampled_imgs = output.column('frame').load()
    downsampled_imgs_npy = np.array([img[1] for img in downsampled_imgs])
    # Create labels numpy array based on time.
    interval = 100
    labels = np.zeros(len(feature_vec_npy))
    for i in range(0, len(labels), interval):
        labels[i: i + interval] = (i // interval)
    labels[i + interval:] = (i // interval) + 1
    print(labels)
    # Write numpy arrays to output directory.
    np.save('%s/feature_vectors.npy' % output_dir, feature_vec_npy)
    np.save('%s/data.npy' % output_dir, downsampled_imgs_npy)
    np.save('%s/labels.npy' % output_dir, labels)
