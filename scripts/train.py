import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import numpy
import sys

import utils
from model import ACModel


# Parse arguments

parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")

## Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

## Parameters for memory
parser.add_argument("--mem_type", type=str, default='lstm',
                    help="memory type: lstm | trxl | trxli | \
                    gtrxl-input | gtrxl-output | gtrxl-highway | \
                    gtrxl-sigmoidtanh | gtrxl-gru")
parser.add_argument("--n_layer", type=int, default=5, help="TrXL layer num")
parser.add_argument("--n_head", type=int, default=8, help="TrXL head num")
parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
parser.add_argument("--mem_len", type=int, default=20, help="memory length")


## Parameters for dreamer
parser.add_argument("--beta_rep_kl", type=float, default=0.1, help="beta for KL term")
parser.add_argument("--n_imagine", type=int, default=5, help="the number of imaginary step")
parser.add_argument("--loss_type", type=str, default='agent-rep-img',
                    help="combination of agent, rep and img")
parser.add_argument("--combine_loss", type=int, default=0, help="whether train rep NN with agent loss or not")
parser.add_argument("--lr_rep", type=float, default=0.001, help="learning rate for representation")
parser.add_argument("--lr_img", type=float, default=8e-5, help="learning rate for imagination")
parser.add_argument("--use_real", type=int, default=1, help="use real trajectories at the first imagination")
parser.add_argument("--img_epochs", type=int, default=5, help="iteration to train the policies")
parser.add_argument("--rep_epochs", type=int, default=50, help="iteration to train the representation layers")

## Visualize
parser.add_argument("--visualize", type=int, default=0, help="visualization or not")
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes to visualize")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")

args = parser.parse_args()

if 'trxl' in args.mem_type:
    args.mem = True
else:
    args.mem = args.recurrence > 1

if args.combine_loss == 1:
    args.combine_loss = True
else:
    args.combine_loss = False

if args.use_real == 1:
    args.use_real = True
else:
    args.use_real = False

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
if args.mem:
    default_model_name = f"{args.env}_{args.algo}_{args.loss_type}_Comb{args.combine_loss}_{args.mem_type}_"

    if args.mem_type == 'lstm':
        default_model_name += f"Rec{args.recurrence}_"
    else:
        default_model_name += f"Nlayer{args.n_layer}_MemLen{args.mem_len}_"

    if args.algo == 'dreamer':
        default_model_name += f"LrRep{args.lr_rep}_LrImg{args.lr_img}_Nimg{args.n_imagine}_UseReal{args.use_real}_"
        default_model_name += f"ImgEpochs{args.img_epochs}_RepEpochs{args.rep_epochs}_"
    else:
        default_model_name += f"Lr{args.lr}_RepEpochs{args.rep_epochs}_"

    default_model_name += f"FPP{args.frames_per_proc}_seed{args.seed}_{date}"

else:
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}"

if args.model == 'None' or args.model is None:
    model_name = default_model_name
else:
    model_name = args.model
model_dir = utils.get_model_dir(model_name)

# Load loggers and Tensorboard writer

txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log command and all script arguments

txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# Set seed for all randomness sources

utils.seed(args.seed)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
txt_logger.info(f"Device: {device}\n")

# Load environments

envs = []
for i in range(args.procs):
    envs.append(utils.make_env(args.env, args.seed + 10000 * i))
txt_logger.info("Environments loaded\n")

# Load training status

try:
    status = utils.get_status(model_dir)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# Load observations preprocessor

obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# Load model

acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text,
                  args.mem_type, args.n_layer, args.n_head, args.dropout, args.mem_len,
                  args.beta_rep_kl, args.n_imagine, args.loss_type, args.combine_loss)
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

# Visualize

if args.visualize:
    agent = utils.Agent(preprocess_obss, acmodel, args.mem_type, model_dir=model_dir,
                        argmax=args.argmax, device=device)

    if args.gif:
       from array2gif import write_gif
       frames = []

    # Create a window to view the environment
    #envs[0].render()
    print('Visualization start')

    for episode in range(args.episodes):
        print("Episode: ",episode, args.episodes)
        obs = envs[0].reset()

        while True:
            #envs[0].render()
            #if args.gif:
            #    frames.append(numpy.moveaxis(envs[0].render("rgb_array"), 2, 0))

            action, est_reward = agent.get_action(obs)

            if args.gif:
                img = envs[0].render("rgb_array")
                H, W, C = img.shape
                if 'GoodObject' in args.env:
                    est_reward = (numpy.clip(est_reward,-1,1)+1)/2
                else:
                    est_reward = numpy.clip(est_reward,0,1)
                color_bar = numpy.ones((H,10,C), dtype=int) * int(est_reward*255)
                img = numpy.concatenate((img, color_bar), axis=1)
                frames.append(numpy.moveaxis(img, 2, 0))
                #frames.append(numpy.moveaxis(envs[0].render("rgb_array"), 2, 0))

            obs, reward, done, _ = envs[0].step(action)
            agent.analyze_feedback(reward, done)

            #if done or envs[0].window.closed:
            if done:
                break

        #if envs[0].window.closed:
        #    break

    if args.gif:
        print("Saving gif... ", end="")
        write_gif(numpy.array(frames), "gifs/"+args.gif+".gif", fps=1/args.pause)
        print("Done.")
    print("Visualization done.")
    exit(1)

# Load algo

if args.algo == "a2c":
    raise NotImplementedError("Not implemented correctly yet")
    algo = torch_ac.A2CAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss,
                            mem_type=args.mem_type, mem_len=args.mem_len, n_layer=args.n_layer, loss_type=args.loss_type)
elif args.algo == "ppo":
    algo = torch_ac.PPOAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                            mem_type=args.mem_type, mem_len=args.mem_len, n_layer=args.n_layer, loss_type=args.loss_type,
                            combine_loss=args.combine_loss, lr_rep=args.lr_rep, rep_epochs=args.rep_epochs)
elif args.algo == "dreamer":
    algo = torch_ac.DREAMERAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                args.optim_eps, args.clip_eps, args.batch_size, preprocess_obss,
                                mem_type=args.mem_type, mem_len=args.mem_len, n_layer=args.n_layer,
                                loss_type=args.loss_type, combine_loss=args.combine_loss,
                                lr_rep=args.lr_rep, lr_img=args.lr_img, n_imagine=args.n_imagine, use_real=args.use_real,
                                img_epochs=args.img_epochs, rep_epochs=args.rep_epochs)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters

    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)
    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs

    if update % args.log_interval == 0:
        fps = logs["num_frames"]/(update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        header = ["update", "frames", "FPS", "duration"]
        data = [update, num_frames, fps, duration]
        header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
        data += rreturn_per_episode.values()
        header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
        data += num_frames_per_episode.values()
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        header += ["rep_loss_total", "rep_loss_recon_acc", "rep_loss_recon", "rep_loss_recon_col", "rep_loss_recon_obj",
            "rep_loss_recon_state", "rep_loss_reward", "rep_loss_reward_nonzero",
            "rep_loss_reward_zero", "rep_loss_reward_nonzero_num", "rep_loss_reward_zero_num",
            "rep_loss_kl"]
        data += [logs["rep_loss"], logs["recon_acc"], logs["recon_loss"], logs["recon_col_loss"],
            logs["recon_obj_loss"], logs["recon_state_loss"], logs["reward_loss"],
            logs["nonzero_reward_loss"], logs["zero_reward_loss"],
            logs["nonzero_reward_num"], logs["zero_reward_num"],
            logs["kl_loss"]]

        txt_logger.info(
            "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
            .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

    # Save status

    if args.save_interval > 0 and update % args.save_interval == 0:
        if args.algo == 'dreamer':
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(),
                      "rep_optimizer_state": algo.rep_optimizer.state_dict(),
                      "img_optimizer_state": algo.img_optimizer.state_dict()}
        else:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(), "ppo_optimizer_state": algo.agent_optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")
