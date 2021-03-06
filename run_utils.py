import sys
import time
sys.stdout=sys.stderr
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

from language_model import LM
from common import CheckpointLoader,print_debug

def run_train(dataset, hps, logdir, ps_device, task=0, master=""):
    with tf.variable_scope("model"):
        print_debug('loading LM model')
        model = LM(hps, "train", ps_device)
    stime = time.time()
    print("Current time: %s" % stime)
    print("ALL VARIABLES")
    for v in tf.all_variables():
        print("%s %s %s %s" % (v.name, v.get_shape(), v.dtype, v.device))
    print("TRAINABLE VARIABLES")
    for v in tf.trainable_variables():
        print("%s %s %s %s" % (v.name, v.get_shape(), v.dtype, v.device))
    print("LOCAL VARIABLES")
    for v in tf.local_variables():
        print("%s %s %s %s" % (v.name, v.get_shape(), v.dtype, v.device))

    sv = tf.train.Supervisor(is_chief=(task == 0),
                             logdir=logdir, # logdir=None, # logdir=logdir,
                             summary_op=None,  # Automatic summaries don't work with placeholders.
                             global_step=model.global_step,
                             save_summaries_secs=60*hps.save_summary_every_min,
                             save_model_secs=60*hps.save_model_every_min)
                             #save_summaries_secs=30,
                             #save_model_secs=120 * 5)

    #config = tf.ConfigProto(allow_soft_placement=True,
    #                        intra_op_parallelism_threads=2,
    #                        inter_op_parallelism_threads=20)
    config = tf.ConfigProto(allow_soft_placement=True)
    close_summary_writer = False
    with sv.managed_session(master, config=config, start_standard_services=True, close_summary_writer=False) as sess:

        # Slowly increase the number of workers during beginning of the training.
        #while not sv.should_stop() and (time.time() - stime) < hps.max_time:
        #    step = int(sess.run(model.global_step))
        #    waiting_until_step = task * hps.num_delayed_steps
        #    if step >= waiting_until_step:
        #        break
        #    else:
        #        print("Current step is %d. Waiting until: %d" % (step, waiting_until_step))
        #    time.sleep(20.0)

        local_step = 0
        prev_global_step = sess.run(model.global_step)
        cur_global_step = 0
        prev_time = time.time()
        data_iterator = dataset.iterate_forever(hps.batch_size * hps.num_gpus, hps.num_steps)

        print_debug('before looping model, sv.save_path=%s , sv.should_stop()=%d, (time.time() - stime)=%.2fs, hps.max_time=%.2fs ' %(sv.save_path, sv.should_stop(), (time.time() - stime), hps.max_time))

        while not sv.should_stop() and (time.time() - stime) < hps.max_time:
            if (int(time.time()) - int(stime)) % 10 == 0:
                print_debug('While In looping model, sv.should_stop()=%d, (time.time() - stime)=%.2fs, hps.max_time=%.2fs ' %(sv.should_stop(), (time.time() - stime), hps.max_time))

            fetches = [model.global_step, model.loss, model.train_op]
            # Chief worker computes summaries every 100 steps.
            should_compute_summary = (task == 0  and local_step % 100 == 0)
            if should_compute_summary:
                fetches += [model.summary_op]

            #x, y, w = next(data_iterator)
            x, y = next(data_iterator)
            should_run_profiler = (hps.run_profiler and task == 0 and local_step % 1000 == 13)
            if should_run_profiler:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                #fetched = sess.run(fetches, {model.x: x, model.y: y, model.w: w},
                fetched = sess.run(fetches, {model.x: x, model.y: y},
                                   options=run_options, run_metadata=run_metadata)
                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                print("Running profiler")
                with open(logdir + "/timeline.json", 'w') as f:
                    f.write(ctf)
                print("Finished profiling!")
            else:
                #fetched = sess.run(fetches, {model.x: x, model.y: y, model.w: w})
                fetched = sess.run(fetches, {model.x: x, model.y: y})
            
            cur_global_step = fetched[0]

            local_step += 1
            if should_compute_summary:
                #print_debug('should_compute_summary!!! BUT WE DROPED THIS MODE TO SAVE MEMORY SPACE sv.should_stop()=%d, (time.time() - stime)=%.2fs, hps.max_time=%.2fs ' %(sv.should_stop(), (time.time() - stime), hps.max_time))
                sv.summary_computed(sess, fetched[-1])

            if local_step < 10 or local_step % 20 == 0:
                cur_time = time.time()
                num_words = hps.batch_size * hps.num_gpus * hps.num_steps
                wps = (cur_global_step - prev_global_step) * num_words / (cur_time - prev_time)
                prev_global_step = cur_global_step
                print("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    cur_global_step, cur_time - prev_time, wps, fetched[1]))
                prev_time = cur_time
        #save last model
        print_debug('Supervisor Begin Save after training period')
        sv._saver.save(sess, sv.save_path, cur_global_step)
        print_debug('Supervisor DONE Save after training period')

    # close sv  with close summery flag 
    sv.stop(None, close_summary_writer)
    sess.close()
    tf.reset_default_graph()


def run_eval(dataset, hps, logdir, mode, num_eval_steps):
    print_debug('run_eval logdir=%s ' % (logdir))

    with tf.variable_scope("model"):
        hps.num_sampled = 0  # Always using full softmax at evaluation.
        hps.keep_prob = 1.0
        #model = LM(hps, "eval", "/cpu:0")
        model = LM(hps, "eval", "/gpu:0")

    if hps.average_params:
        print("Averaging parameters for evaluation.")
        saver = tf.train.Saver(model.avg_dict)
    else:
        saver = tf.train.Saver()

    # Use only 4 threads for the evaluation.
    #config = tf.ConfigProto(allow_soft_placement=True,
    #                        intra_op_parallelism_threads=20,
    #                        inter_op_parallelism_threads=1)
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    sw = tf.summary.FileWriter(logdir + "/" + mode, sess.graph)
    print_debug('run_eval tf.summary.FileWriter=%s ' % (logdir + "/" + mode))

    ckpt_loader = CheckpointLoader(saver, model.global_step, logdir + "/train")
    print_debug('run_eval ckpt_loader=%s ' % (ckpt_loader.logdir))

    with sess.as_default():
        print_debug('run_eval sess.as_default iteration')

        while ckpt_loader.load_checkpoint():
            print_debug('eval load_checkpoint chunk Loader done!')

            global_step = ckpt_loader.last_global_step
            if mode == "eval_full":
                data_iterator = dataset.iterate_forever(hps.batch_size*hps.num_gpus,hps.num_steps)
            else:
                data_iterator = dataset.iterate_once(hps.batch_size*hps.num_gpus,hps.num_steps)

            print_debug('eval run local variables initalizer')

            #tf.initialize_local_variables().run()
            tf.local_variables_initializer().run()
            loss_nom = 0.0
            loss_den = 0.0

            print_debug('eval run for loop of enumerated data iterator mode='+mode+' eval_steps='+str(num_eval_steps))
            #for i, (x, y, w) in enumerate(data_iterator):
            for i, (x, y) in enumerate(data_iterator):
                if i >= num_eval_steps and mode!="eval_full":
                    break
                #loss = sess.run(model.loss, {model.x: x, model.y: y, model.w: w})
                loss = sess.run(model.loss, {model.x: x, model.y: y})



                loss_nom += loss
                loss_den += 1 # ???
                #loss_den += w.mean()
                loss = loss_nom / loss_den
                #sys.stdout.write("%d: %.3f (%.3f) ... " % (i, loss, np.exp(loss)))
                #sys.stdout.flush()
                #sys.stdout.write("\n")


            log_perplexity = loss_nom / loss_den
            print("Results at %d: log_perplexity = %.3f perplexity = %.3f" % (
                global_step, log_perplexity, np.exp(log_perplexity)))

            summary = tf.Summary()
            summary.value.add(tag='eval/log_perplexity', simple_value=log_perplexity)
            summary.value.add(tag='eval/perplexity', simple_value=np.exp(log_perplexity))
            sw.add_summary(summary, global_step)
            sw.flush()
            #if mode == "eval_full":
            #    break #we don't need to wait for other checkpoints in this mode

            break #we always break

        print_debug('run_eval END OF WHILE loader loop')

    print_debug('run_eval END OF WHILE session loop')
    sess.close()
    tf.reset_default_graph()

def run_statistic(dataset, hps, logdir, ps_device, task=0, master=""):
    with tf.variable_scope("model"):
        print_debug('loading LM model')
        model = LM(hps, "train", ps_device)
    stime = time.time()
    print("Current time: %s" % stime)
    print("ALL VARIABLES")
    for v in tf.all_variables():
        print("%s %s %s %s" % (v.name, v.get_shape(), v.dtype, v.device))
    print("TRAINABLE VARIABLES")
    for v in tf.trainable_variables():
        print("%s %s %s %s" % (v.name, v.get_shape(), v.dtype, v.device))
    print("LOCAL VARIABLES")
    for v in tf.local_variables():
        print("%s %s %s %s" % (v.name, v.get_shape(), v.dtype, v.device))

    sv = tf.train.Supervisor(is_chief=(task == 0),
                             logdir=logdir,  # logdir=None, # logdir=logdir,
                             summary_op=None,  # Automatic summaries don't work with placeholders.
                             #global_step=model.global_step,
                             save_summaries_secs=60 * hps.save_summary_every_min,
                             save_model_secs=60 * hps.save_model_every_min)
    # save_summaries_secs=30,
    # save_model_secs=120 * 5)

    # config = tf.ConfigProto(allow_soft_placement=True,
    #                        intra_op_parallelism_threads=2,
    #                        inter_op_parallelism_threads=20)
    config = tf.ConfigProto(allow_soft_placement=True)
    close_summary_writer = False
    with sv.managed_session(master, config=config, start_standard_services=True, close_summary_writer=False) as sess:

        # Slowly increase the number of workers during beginning of the training.
        # while not sv.should_stop() and (time.time() - stime) < hps.max_time:
        #    step = int(sess.run(model.global_step))
        #    waiting_until_step = task * hps.num_delayed_steps
        #    if step >= waiting_until_step:
        #        break
        #    else:
        #        print("Current step is %d. Waiting until: %d" % (step, waiting_until_step))
        #    time.sleep(20.0)

        local_step = 0
        prev_global_step = sess.run(model.global_step)
        cur_global_step = 0
        prev_time = time.time()
        data_iterator = dataset.iterate_forever(hps.batch_size * hps.num_gpus, hps.num_steps)

        print_debug(
            'before looping model, sv.save_path=%s , sv.should_stop()=%d, (time.time() - stime)=%.2fs, hps.max_time=%.2fs ' % (
            sv.save_path, sv.should_stop(), (time.time() - stime), hps.max_time))

        while not sv.should_stop() and (time.time() - stime) < hps.max_time:
            if (int(time.time()) - int(stime)) % 10 == 0:
                print_debug(
                    'While In looping model, sv.should_stop()=%d, (time.time() - stime)=%.2fs, hps.max_time=%.2fs ' % (
                    sv.should_stop(), (time.time() - stime), hps.max_time))

            fetches = [model.global_step, model.loss, model.train_op]
            # Chief worker computes summaries every 100 steps.
            should_compute_summary = (task == 0 and local_step % 100 == 0)
            if should_compute_summary:
                fetches += [model.summary_op]

            # x, y, w = next(data_iterator)
            x, y = next(data_iterator)
            should_run_profiler = (hps.run_profiler and task == 0 and local_step % 1000 == 13)
            if should_run_profiler:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                # fetched = sess.run(fetches, {model.x: x, model.y: y, model.w: w},
                fetched = sess.run(fetches, {model.x: x, model.y: y},
                                   options=run_options, run_metadata=run_metadata)
                # Create the Timeline object, and write it to a json
                tl = timeline.Timeline(run_metadata.step_stats)
                ctf = tl.generate_chrome_trace_format()
                print("Running profiler")
                with open(logdir + "/timeline.json", 'w') as f:
                    f.write(ctf)
                print("Finished profiling!")
            else:
                # fetched = sess.run(fetches, {model.x: x, model.y: y, model.w: w})
                fetched = sess.run(fetches, {model.x: x, model.y: y})

            cur_global_step = fetched[0]

            local_step += 1
            if should_compute_summary:
                # print_debug('should_compute_summary!!! BUT WE DROPED THIS MODE TO SAVE MEMORY SPACE sv.should_stop()=%d, (time.time() - stime)=%.2fs, hps.max_time=%.2fs ' %(sv.should_stop(), (time.time() - stime), hps.max_time))
                sv.summary_computed(sess, fetched[-1])

            if local_step < 10 or local_step % 20 == 0:
                cur_time = time.time()
                num_words = hps.batch_size * hps.num_gpus * hps.num_steps
                wps = (cur_global_step - prev_global_step) * num_words / (cur_time - prev_time)
                prev_global_step = cur_global_step
                print("Iteration %d, time = %.2fs, wps = %.0f, train loss = %.4f" % (
                    cur_global_step, cur_time - prev_time, wps, fetched[1]))
                prev_time = cur_time
        # save last model
        print_debug('Supervisor Begin Save after training period')
        sv._saver.save(sess, sv.save_path, cur_global_step)
        print_debug('Supervisor DONE Save after training period')

    # close sv  with close summery flag
    sv.stop(None, close_summary_writer)
