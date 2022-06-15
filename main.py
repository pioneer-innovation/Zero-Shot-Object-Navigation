from __future__ import print_function, division

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,7"
import random
import ctypes
import setproctitle
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

from utils import flag_parser

from utils.class_finder import model_class, agent_class, optimizer_class
from utils.net_util import ScalarMeanTracker
from main_eval import main_eval,main_eval_unseen,main_eval_seen

from runners import nonadaptivea3c_train, nonadaptivea3c_val, savn_train, savn_val,savn_train_seen
from runners import nonadaptivea3c_train_seen,nonadaptivea3c_val_unseen


os.environ["OMP_NUM_THREADS"] = "1"


def main():
    setproctitle.setproctitle("Train/Test Manager")
    args = flag_parser.parse_arguments()
    print(args)
    if args.model == "SAVN":
        if args.zsd:
            print("use zsd setting !")
            args.learned_loss = True
            args.num_steps = 6
            target = savn_val if args.eval else savn_train_seen
        else:
            args.learned_loss = True
            args.num_steps = 6
            target = savn_val if args.eval else savn_train
    else:
        if args.zsd:
            print("use zsd setting !")
            args.learned_loss = False
            args.num_steps = 50
            target = nonadaptivea3c_val_unseen if args.eval else nonadaptivea3c_train_seen
        else:
            args.learned_loss = False
            args.num_steps = 50
            target = nonadaptivea3c_val if args.eval else nonadaptivea3c_train

    create_shared_model = model_class(args.model)
    init_agent = agent_class(args.agent_type)
    optimizer_type = optimizer_class(args.optimizer)

    if args.eval:
        if args.zsd:
            print("Evaluate Unseen Classes !")
            main_eval_unseen(args, create_shared_model, init_agent)
            print("#######################################################")
            print("Evaluate Seen Classes !")
            main_eval_seen(args, create_shared_model, init_agent)
            return
        else:
            main_eval(args, create_shared_model, init_agent)
            return

    start_time = time.time()
    local_start_time_str = time.strftime(
        "%Y-%m-%d_%H:%M:%S", time.localtime(start_time)
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.log_dir is not None:
        tb_log_dir = args.log_dir + "/" + args.title + "-" + local_start_time_str
        log_writer = SummaryWriter(log_dir=tb_log_dir)
    else:
        log_writer = SummaryWriter(comment=args.title)

    if args.gpu_ids == -1:
        args.gpu_ids = [-1]
    else:
        torch.cuda.manual_seed(args.seed)
        mp.set_start_method("spawn")

    shared_model = create_shared_model(args)

    train_total_ep = 0
    n_frames = 0

    if shared_model is not None:
        shared_model.share_memory()
        optimizer = optimizer_type(
            filter(lambda p: p.requires_grad, shared_model.parameters()), args
        )
        optimizer.share_memory()
        print(shared_model)
    else:
        assert (
            args.agent_type == "RandomNavigationAgent"
        ), "The model is None but agent is not random agent"
        optimizer = None

    processes = []

    end_flag = mp.Value(ctypes.c_bool, False)

    train_res_queue = mp.Queue()

    for rank in range(0, args.workers):
        p = mp.Process(
            target=target,
            args=(
                rank,
                args,
                create_shared_model,
                shared_model,
                init_agent,
                optimizer,
                train_res_queue,
                end_flag,
            ),
        )
        p.start()
        processes.append(p)
        time.sleep(0.1)

    print("Train agents created.")

    train_thin = args.train_thin
    train_scalars = ScalarMeanTracker()

    save_entire_model = 0
    try:
        time_start = time.time()
        reward_avg_list = []
        ep_length_avg_list=[]
        while train_total_ep < args.max_ep:
            train_result = train_res_queue.get()
            reward_avg_list.append(train_result["total_reward"])
            train_scalars.add_scalars(train_result)
            train_total_ep += 1
            n_frames += train_result["ep_length"]
            ep_length_avg_list.append(train_result["ep_length"])
            if (train_total_ep % train_thin) == 0:
                log_writer.add_scalar("n_frames", n_frames, train_total_ep)
                tracked_means = train_scalars.pop_and_reset()
                for k in tracked_means:
                    log_writer.add_scalar(
                        k + "/train", tracked_means[k], train_total_ep
                    )

            if (train_total_ep % args.ep_save_freq) == 0:
                print(n_frames)
                if not os.path.exists(args.save_model_dir):
                    os.makedirs(args.save_model_dir)
                state_to_save = shared_model.state_dict()
                save_path = os.path.join(
                    args.save_model_dir,
                    "{0}_{1}_{2}_{3}.dat".format(
                        args.model, train_total_ep, n_frames, local_start_time_str
                    ),
                )
                torch.save(state_to_save, save_path)

                save_entire_model += 1
                if (save_entire_model % 5) == 0:
                    state = {
                        'epoch': train_total_ep,
                        'state_dict': shared_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }

                    save_model_path = os.path.join(
                        args.save_model_dir,
                        "{0}_{1}_{2}.tar".format(
                            args.model, train_total_ep, local_start_time_str
                        ),
                    )
                    torch.save(state,save_model_path)
                    save_entire_model=0

            if train_total_ep % 100 == 0:
                time_end = time.time()
                seconds = round(time_end - time_start)
                m, s = divmod(seconds, 60)
                h, m = divmod(m, 60)
                reward_avg = sum(reward_avg_list)/len(reward_avg_list)
                ep_length_avg = sum(ep_length_avg_list)/len(ep_length_avg_list)
                print("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]"
                      .format(train_total_ep, args.max_ep, h, m, s,reward_avg,ep_length_avg))
                reward_avg_list = []
                ep_length_avg_list = []
                save_path = os.path.join( args.save_model_dir,"{0}_{1}.txt".format(args.model,local_start_time_str))
                f = open(save_path, "a")
                if train_total_ep == 100:
                    f.write(str(args))
                    f.write("\n")
                    f.write("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]\n"
                            .format(train_total_ep, args.max_ep, h, m, s, reward_avg, ep_length_avg))
                else:
                    f.write("epoch:[{:d}]/[{:d}] time:[{:02d}:{:02d}:{:02d}] reward:[{:.2f}] ep_length:[{:.1f}]\n"
                          .format(train_total_ep, args.max_ep, h, m, s,reward_avg,ep_length_avg))
                f.close()
    finally:
        log_writer.close()
        end_flag.value = True
        for p in processes:
            time.sleep(0.1)
            p.join()


if __name__ == "__main__":
    main()
