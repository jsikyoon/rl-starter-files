import argparse
import time
import datetime
import torch
import torch_ac
import tensorboardX
import sys

import utils
from model import ACModel

# For unity
from pyvirtualdisplay import Display


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
parser.add_argument("--frames", type=int, default=10**8,
                    help="number of frames of training (default: 1e8)")

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
parser.add_argument("--attn_type", type=int, default=2, help="attention type")
parser.add_argument("--n_layer", type=int, default=5, help="TrXL layer num")
parser.add_argument("--n_head", type=int, default=8, help="TrXL head num")
parser.add_argument("--ext_len", type=int, default=20, help="the length of given input that is not target")
parser.add_argument("--mem_len", type=int, default=20, help="memory length")

## Parameters for image encoder
parser.add_argument("--img_encode", type=int, default=0, help="Using image or compact encoding")

## Parameters for VMPO
parser.add_argument("--alpha", type=float, default=0.1, help="VMPO hyperparameter")
parser.add_argument("--T_target", type=int, default=10, help="Target update period")

## Parameters for evaluation
parser.add_argument("--eval_procs", type=int, default=20,
                    help="number of evaluation processes (default: 20)")

args = parser.parse_args()

if 'trxl' in args.mem_type:
  args.mem = True
else:
  args.mem = args.recurrence > 1

args.img_encode = True if args.img_encode==1 else False

if args.env.split('-')[0] == 'Unity':
    assert args.img_encode==True

# Set run dir

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.algo}_ImgEncode{args.img_encode}"
if args.algo == 'vmpo':
    default_model_name += f"_alpha{args.alpha}_T{args.T_target}_Epoch{args.epochs}"
elif args.algo == 'ppo':
    default_model_name += f"_Epoch{args.epochs}_EntropyCoef{args.entropy_coef}"
if args.mem:
    default_model_name += f"_{args.mem_type}_Rec{args.recurrence}"
    if 'trxl' in args.mem_type:
        default_model_name += f"_AttnType{args.attn_type}_Nlayer{args.n_layer}_ExtLen{args.ext_len}_MemLen{args.mem_len}"
default_model_name += f"_Lr{args.lr}_seed{args.seed}_{date}"

model_name = args.model or default_model_name
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

if args.env.split('-')[0] == 'Unity':
    display = Display(backend='xvnc', size=(64, 64), visible=0, rfbport=0)
    display.start()
envs = []
for i in range(args.procs):
    envs.append(utils.make_env(args.env, args.img_encode, args.seed + 10000 * i))
eval_envs = []
for i in range(args.eval_procs):
    eval_envs.append(utils.make_env(args.env, args.img_encode, args.seed + 10000 * (args.procs+i)))
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
                  args.mem_type, args.attn_type, args.n_layer, args.n_head, args.ext_len, args.mem_len,
                  args.img_encode)
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

# Load algo

if args.algo == "a2c":
    algo = torch_ac.A2CAlgo(eval_envs, envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_alpha, args.optim_eps, preprocess_obss,
                            mem_type=args.mem_type, ext_len=args.ext_len, mem_len=args.mem_len, n_layer=args.n_layer)
elif args.algo == "ppo":
    algo = torch_ac.PPOAlgo(eval_envs, envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                            mem_type=args.mem_type, ext_len=args.ext_len, mem_len=args.mem_len, n_layer=args.n_layer)
elif args.algo == "vmpo":
    acmodel_learner = ACModel(obs_space, envs[0].action_space, args.mem, args.text,
                              args.mem_type, args.attn_type, args.n_layer, args.n_head, args.ext_len, args.mem_len,
                              args.img_encode).to(device)
    acmodel_learner.load_state_dict(acmodel.state_dict())
    algo = torch_ac.VMPOAlgo(eval_envs, envs, acmodel, acmodel_learner, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss,
                            mem_type=args.mem_type, ext_len=args.ext_len, mem_len=args.mem_len, n_layer=args.n_layer,
                            alpha=args.alpha, T_target=args.T_target)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
txt_logger.info("Optimizer loaded\n")

# Train model

num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

done = False
done_cnt = 0
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
        # evaluation
        header += ["eval_return"]
        eval_return = algo.run_evaluation()
        data += [eval_return]
        if args.algo == 'vmpo':
          header += ["log_prob", "value", "policy_loss", "value_loss", "grad_norm"]
          data += [logs["log_prob"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

          txt_logger.info(
              "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | eR {:.2f} | Lp {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
              .format(*data))
        else:
          header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
          data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

          txt_logger.info(
              "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | eR {:.2f} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
              .format(*data))

        header += ["return_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()

        # evaluation
        header += ["eval_return"]
        data += [algo.run_evaluation()]

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            tb_writer.add_scalar(field, value, num_frames)

        if 'IMaze' in args.env:
            if eval_return > 10: # average return is over 10
                done_cnt += 1
            else:
                done_cnt = 0
            if done_cnt >= 10: # success over 10 intervals
                done = True

    # Save status

    if (args.save_interval > 0 and update % args.save_interval == 0) or done:
        status = {"num_frames": num_frames, "update": update,
                  "model_state": acmodel.state_dict(), "optimizer_state": algo.optimizer.state_dict()}
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir)
        txt_logger.info("Status saved")

    if done:
        break

for _env in envs:
    _env.close()
for _env in eval_envs:
    _env.close()
if args.env.split('-')[0] == 'Unity':
    display.stop()
