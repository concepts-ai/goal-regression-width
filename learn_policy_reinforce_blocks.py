
import collections
import copy
import functools
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import jacinle.random as random
import jacinle.io as io
from gregress_theory.cli import format_args
from gregress_theory.envs import BlockWorldDisplayer
from gregress_theory.envs import make as make_env
from gregress_theory.nn.neural_logic import LogicMachine, InputTransform, LogicInference, LogitsInference
from gregress_theory.nn.neural_logic.modules._utils import meshgrid_exclude_self
from gregress_theory.nn.rl.reinforce import REINFORCELoss
from gregress_theory.thutils import monitor_gradrms
from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger, set_output_file
from jacinle.utils.container import GView
from jacinle.utils.meter import GroupMeters
from jacinle.utils.tqdm import tqdm_pbar
from jactorch.optim.accum_grad import AccumGrad
from jactorch.optim.quickaccess import get_optimizer
from jactorch.train.env import TrainerEnv
from jactorch.utils.meta import as_cuda, as_numpy, as_tensor

TASKS = ['single-clear']

parser = JacArgumentParser()

parser.add_argument('--concat-worlds', action='store_true')
parser.add_argument('--attributes', type=int, default=8, metavar='N')
parser.add_argument('--pred-depth', type=int, default=None, metavar='N')
LogicMachine.make_prog_block_parser(parser, {
    'depth': 3,
    'breadth': 2,
    'exclude_self': True,
    'logic_model': 'mlp',
    'logic_hidden_dim': []
})

parser.add_argument('--hidden-size', type=int, default=64, metavar='N')
parser.add_argument('--num-repeats', type=int, default=4, metavar='N')
parser.add_argument('--key-length', type=int, default=16, metavar='N')
parser.add_argument('--value-length', type=int, default=32, metavar='N')
parser.add_argument('--code-length', type=int, default=8, metavar='N')
parser.add_argument('--threshold', type=float, default=0.99, metavar='F')

parser.add_argument('--blocks', type=int, default=5, metavar='N')
parser.add_argument('--test-blocks', type=int, default=50, metavar='N')
parser.add_argument('--test-blocks-interval', type=int, default=10, metavar='N')

parser.add_argument('--task', required=True, choices=TASKS)
parser.add_argument('--curriculum', action='store_true')
parser.add_argument('--enable-mining', action='store_true')
parser.add_argument('--prob-sample-failure', type=float, default=0.1, metavar='F')
parser.add_argument('--remove-failure-argmax', action='store_true')
parser.add_argument('--curriculum-count', type=int, default=3, metavar='N')
parser.add_argument('--graduate', type=int, default=12, metavar='N')
parser.add_argument('--final-exam', action='store_true')

parser.add_argument('--use-gpu', action='store_true')
parser.add_argument('--optimizer', default='AdamW', choices=['SGD', 'Adam', 'AdamW'])
parser.add_argument('--lr', type=float, default=0.005, metavar='F')
parser.add_argument('--gamma', type=float, default=0.99, metavar='F')
parser.add_argument('--penalty', type=float, default=-0.01, metavar='F')
parser.add_argument('--pred-weight', type=float, default=0.1, metavar='F')
parser.add_argument('--entropy-beta', type=float, default=0.01, metavar='F')
parser.add_argument('--temp', type=float, default=6.0, metavar='F')
parser.add_argument('--temp-lower-bound', type=float, default=1.0, metavar='F')
parser.add_argument('--temp-decay', type=float, default=0.993, metavar='F')
parser.add_argument('--sample-nr-blocks', action='store_true')
parser.add_argument('--sample-blocks-maxlen', type=int, default=3, metavar='N')
parser.add_argument('--batch-size', type=int, default=4, metavar='N')
parser.add_argument('--accum-grad', type=int, default=1, metavar='N')
parser.add_argument('--epoch-size', type=int, default=20, metavar='N')
parser.add_argument('--replay-epoch-size', type=int, default=10, metavar='N')
parser.add_argument('--epochs', type=int, default=1000, metavar='N')

parser.add_argument('--test-epoch-size', type=int, default=10, metavar='N')
parser.add_argument('--random-order', action='store_true')
parser.add_argument('--visualize-times', type=int, default=0, metavar='N')

parser.add_argument('--seed', type=int, default=None, metavar='SEED')

parser.add_argument('--dump-dir', default=None, metavar='DIR')
parser.add_argument('--dump-play', action='store_true')
parser.add_argument('--load-checkpoint', default=None, metavar='FILE')
parser.add_argument('--save-interval', type=int, default=10, metavar='N')
parser.add_argument('--test-interval', type=int, default=500, metavar='N')

args = parser.parse_args()

args.use_gpu = args.use_gpu and torch.cuda.is_available()
args.dump_play = (args.dump_dir is not None)

args.is_two_worlds_tasks = args.task in TASKS[5:]

if args.dump_dir is not None:
    io.mkdir(args.dump_dir)
    args.checkpoints_dir = os.path.join(args.dump_dir, 'checkpoints')
    io.mkdir(args.checkpoints_dir)
    args.log_file = os.path.join(args.dump_dir, 'log.log')
    set_output_file(args.log_file)

if args.seed is not None:
    import jacinle.random as random
    random.reset_global_seed(args.seed)

make_env = functools.partial(make_env, random_order=args.random_order, exclude_self=True)
if args.dump_dir is not None and args.dump_play and args.is_two_worlds_tasks:
    make_env = functools.partial(make_env, fix_ground=True)

logger = get_logger(__file__)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # input_dim = 4 if args.is_two_worlds_tasks else 2
        # self.transform = InputTransform('cmp', exclude_self=False)
        # current_dim = transformed_dim = self.transform.get_output_dim(input_dim)
        # self.feature_dim = 1 if args.concat_worlds else 2
        self.feature_dim = 2
        transformed_dim = 4
        current_dim = 4

        input_dims = [0 for _ in range(args.breadth + 1)]
        input_dims[2] = current_dim
        self.features = LogicMachine.from_args(input_dims, args.attributes, args)
        current_dim = self.features.output_dims[self.feature_dim]

        self.final_transform = InputTransform('concat', exclude_self=False)
        if args.concat_worlds:
            current_dim = self.final_transform.get_output_dim(current_dim) + transformed_dim
            if args.is_two_worlds_tasks:
                current_dim *= 2

        self.pred_valid = LogicInference('mlp', current_dim, 1, [])
        self.pred = LogitsInference('mlp', current_dim, 1, [])
        self.loss = REINFORCELoss(entropy_beta=args.entropy_beta)
        self.pred_loss = nn.BCELoss()

    def forward(self, feed_dict):
        feed_dict = GView(feed_dict)

        states = feed_dict.states.float()
        f = self.get_binary_relations(states)
        logits = self.pred(f).squeeze(dim=-1).view(states.size(0), -1)
        policy = F.softmax(logits, dim=-1)

        if self.training:
            pred_states = feed_dict.pred_states.float()
            f = self.get_binary_relations(pred_states, depth=args.pred_depth)
            f = self.pred_valid(f).squeeze(dim=-1).view(pred_states.size(0), -1)
            valid = f[range(pred_states.size(0)), feed_dict.pred_actions].clamp(min=1e-20)

        if self.training:
            policy = policy.clamp(min=1e-20)
            loss, monitors = self.loss(policy, feed_dict.actions, feed_dict.discount_rewards)
            pred_loss = self.pred_loss(valid, feed_dict.valid)
            monitors['pred/accuracy'] = feed_dict.valid.eq((valid > 0.5).float()).float().mean()
            loss = loss + args.pred_weight * pred_loss
            return loss, monitors, dict()
        else:
            explore_policy = F.softmax(logits / args.temp, dim=-1)
            return dict(policy=policy, explore_policy=explore_policy)

    def get_binary_relations(self, states, depth=None):
        batch_size, total = states.size()[:2]
        # f = self.transform(states)
        f = states
        if args.memory_net:
            f = self.feature(f)
        else:
            inp = [None for i in range(args.breadth + 1)]
            inp[2] = f
            features = self.features(inp, depth=depth)
            f = features[self.feature_dim]

        if args.concat_worlds:
            if args.is_two_worlds_tasks:
                assert total % 2 == 0
                nr_objects = total // 2
                f = torch.cat([f[:, :nr_objects], f[:, nr_objects:]], dim=-1)
                states = torch.cat([states[:, :nr_objects], states[:, nr_objects:]], dim=-1)
            transformed_input = self.transform(states)
            f = torch.cat([self.final_transform(f), transformed_input], dim=-1)
        elif args.is_two_worlds_tasks:
            assert total % 2 == 0
            nr_objects = total // 2
            f = f[:, :nr_objects, :nr_objects].contiguous()

        f = meshgrid_exclude_self(f)
        return f


class ValidMoveDataset(object):
    def __init__(self, maxlen=5000):
        super().__init__()
        self.data = [[collections.deque(maxlen=maxlen) for i in range(2)] for j in range(30)]
        self.blocks = 0

    def append(self, state, action, valid):
        nr_blocks = state.shape[0]
        if args.is_two_worlds_tasks:
            nr_blocks = nr_blocks // 2
        valid = int(valid)
        self.data[nr_blocks][valid].append((state, action))
        self.blocks = max(self.blocks, nr_blocks)

    def _sample(self, data, num, label):
        # assert num <= len(data)
        states, actions = [], []
        for i in range(num):
            ind = random.randint(len(data))
            state, action = data[ind]
            states.append(state)
            actions.append([action])
        return np.array(states), np.array(actions), np.ones((num, )) * label

    def sample_batch(self, batch_size):
        # balance
        data = self.data[self.blocks]
        num = batch_size // 2
        c = 1 - int(len(data[0]) > 0)
        states1, actions1, labels1 = self._sample(data[c], num, c)
        c = int(len(data[1]) > 0)
        states2, actions2, labels2 = self._sample(data[c], batch_size - num, c)
        return np.vstack([states1, states2]), np.vstack([actions1, actions2]).squeeze(axis=-1), np.concatenate([labels1, labels2], axis=0)


def make_data(traj, gamma):
    Q = 0
    discount_rewards = []
    for reward in traj['rewards'][::-1]:
        Q = Q * gamma + reward
        discount_rewards.append(Q)
    discount_rewards.reverse()

    traj['states'] = as_tensor(np.array(traj['states']))
    traj['actions'] = as_tensor(np.array(traj['actions']))
    traj['discount_rewards'] = as_tensor(np.array(discount_rewards)).float()
    return traj


def get_curriculum_blocks(passed, curriculum_count, nr_blocks, pr_blocks):
    if passed:
        get_curriculum_blocks.count += 1
    else:
        get_curriculum_blocks.count = 0
    if get_curriculum_blocks.count >= curriculum_count:
        nr_blocks += 1
        pr_blocks.append(nr_blocks)
        get_curriculum_blocks.count = 0
    return nr_blocks


get_curriculum_blocks.count = 0


def save_checkpoint(trainer, name):
    if args.dump_dir is not None:
        checkpoint_file = os.path.join(args.checkpoints_dir, 'checkpoint_{}.pth'.format(name))
        trainer.save_checkpoint(checkpoint_file)


def train(trainer):
    model = trainer.model

    pr_blocks = collections.deque(maxlen=args.sample_blocks_maxlen)
    nr_blocks = args.blocks
    pr_blocks.append(nr_blocks)
    failures = []

    dataset = ValidMoveDataset()
    for i in range(1, 1 + args.epochs):
        meters = GroupMeters()
        # player = make_env(args.task, nr_blocks)
        new_failures = []
        with tqdm_pbar(total=args.epoch_size) as pbar:
            for j in range(args.epoch_size):
                sample_failure = args.enable_mining and len(failures) > 0 and random.rand() < args.prob_sample_failure
                if sample_failure:
                    ind = random.randint(len(failures))
                    player = copy.deepcopy(failures[ind])
                else:
                    if args.sample_nr_blocks:
                        train_blocks = random.choice(pr_blocks)
                        meters.update(train_blocks=train_blocks)
                    else:
                        train_blocks = nr_blocks
                    player = make_env(args.task, train_blocks)
                    player.restart()
                    player_backup = copy.deepcopy(player)

                model.eval()
                succ, score, traj = run_episode(player, model, dataset=dataset, episode_id=j, need_restart=False)

                if args.enable_mining:
                    if sample_failure and succ == 1.0: # succeed
                        del failures[ind]
                    elif not sample_failure and succ < 1.0: # failed
                        new_failures.append(player_backup)

                feed_dict = make_data(traj, args.gamma)
                states, actions, labels = dataset.sample_batch(args.batch_size)
                feed_dict['pred_states'] = as_tensor(states)
                feed_dict['pred_actions'] = as_tensor(actions)
                feed_dict['valid'] = as_tensor(labels).float()
                if args.use_gpu:
                    feed_dict = as_cuda(feed_dict)

                model.train()
                loss, monitors, output_dict, extras = trainer.step(feed_dict)
                meters.update(monitor_gradrms(model))
                meters.update(monitors)
                meters.update(blocks=nr_blocks, succ=succ, train_score=score, loss=loss, length=len(traj['rewards']))
                meters.update(explore_temp=args.temp)

                pbar.set_description('> Training iter={iter}, loss={loss:.4f}, succ={succ:.4f}'.format(
                    iter=j, **meters.val
                ))
                pbar.update()
        failures.extend(new_failures)
        if args.enable_mining:
            meters.update(fails=len(failures))
        logger.info(meters.format_simple('> Train Epoch {:5d}: '.format(i), compressed=False))

        if args.enable_mining and len(failures) > 0:
            model.eval()
            with tqdm_pbar(total=args.replay_epoch_size) as pbar:
                for j in range(args.replay_epoch_size):
                    if len(failures) == 0:
                        break
                    ind = random.randint(len(failures))
                    player = copy.deepcopy(failures[ind])

                    succ, score, traj = run_episode(player, model, episode_id=j, use_argmax=args.remove_failure_argmax, need_restart=False)
                    if succ == 1.0:
                        del failures[ind]
                    pbar.update()

        if args.curriculum:
            count = args.curriculum_count
            # if args.final_exam and nr_blocks == args.graduate:
            #     count *= 2
            passed = meters.avg['succ'] == 1.0
            if args.enable_mining and len(failures) > 0:
                passed = False
            nr_blocks = get_curriculum_blocks(passed, count, nr_blocks, pr_blocks)
            if nr_blocks > args.graduate:
                break
        args.temp = max(args.temp_lower_bound, args.temp * args.temp_decay)

        if i % args.save_interval == 0:
            save_checkpoint(trainer, str(i))
        if i % args.test_interval == 0:
            test(model)


def test(model):
    model.eval()
    if args.test_blocks_interval > 0:
        test_list = range(args.test_blocks, args.blocks - 1, -args.test_blocks_interval)[::-1]
    else:
        test_list = [args.test_blocks]
    for test_blocks in test_list:
        player = make_env(args.task, test_blocks)
        meters = GroupMeters()
        with tqdm_pbar(total=args.test_epoch_size) as pbar:
            for j in range(args.test_epoch_size):
                succ, score, traj, gt_steps = run_episode(player, model, episode_id=j, eval_only=True, use_argmax=True, gt_steps=True)
                avg_length = len(traj['rewards'])
                meters.update(blocks=test_blocks, succ=succ, score=score, length=avg_length, gt_steps=gt_steps)
                pbar.set_description('> Test iter={iter}, blocks={blocks}, succ={succ}, score={score:.4f}, length={length}'.format(
                    iter=j, **meters.val
                ))
                pbar.update()
        logger.info(meters.format_simple('> Evaluation: ', compressed=False))


def visualize(model):
    env = make_env(args.task, args.test_blocks)
    is_over = False
    step = 1
    decorated = args.is_two_worlds_tasks
    displayer = BlockWorldDisplayer(max_blocks=60, decorated=decorated)
    env.restart()
    while not is_over:
        feed_dict = dict(states=np.array([env.current_state]))
        feed_dict = as_tensor(feed_dict)
        if args.use_gpu:
            feed_dict = as_cuda(feed_dict)

        with torch.no_grad():
            output_dict = model(feed_dict)

        policy = output_dict['policy']
        p = as_numpy(policy.data[0])
        # action = random.choice(len(p), p=p)
        action = p.argmax()
        real_action = map(displayer.get_display_char, env.map_action(action))

        displayer.display(env.current_state, extra='step: {}; action: MOVE({}, {})'.format(step, *real_action), cls=True)

        reward, is_over = env.action(action)
        step += 1

        import time; time.sleep(0.3)

    displayer.display(env.current_state, extra='step: {}; game is over'.format(step))


def run_episode(env, model, dataset=None, episode_id=None, eval_only=False, use_argmax=False, need_restart=True, gt_steps=False):
    is_over = False
    traj = collections.defaultdict(list)
    score = 0
    if need_restart:
        env.restart()
    if gt_steps:
        groundtruth_steps = env.unwrapped.get_groundtruth_steps()
    dump_play = args.dump_play and eval_only
    if dump_play:
        array = env.unwrapped.current_state
        moves = []
        new_pos = []
    while not is_over:
        state = env.current_state
        feed_dict = dict(states=np.array([state]))
        feed_dict = as_tensor(feed_dict)
        if args.use_gpu:
            feed_dict = as_cuda(feed_dict)

        with torch.set_grad_enabled(not eval_only):
            output_dict = model(feed_dict)

        policy = output_dict['explore_policy']
        p = as_numpy(policy.data[0])
        action = p.argmax() if use_argmax else random.choice(len(p), p=p)
        mapped_x, mapped_y = env._mapping[action]
        valid = env.unwrapped.world.moveable(mapped_x, mapped_y)
        reward, is_over = env.action(action)
        if dump_play:
            moves.append([mapped_x, mapped_y])
            res = tuple(env.current_state[mapped_x][2:])
            new_pos.append((int(res[0]), int(res[1])))

        if reward == 0 and args.penalty is not None:
            reward = args.penalty
        succ = 1 if is_over and reward > 0.99 else 0 # assume return 1 only when succeed
        score += reward
        traj['states'].append(state)
        traj['rewards'].append(reward)
        traj['actions'].append(action)
        if not eval_only and dataset is not None and mapped_x != mapped_y:
            dataset.append(state, action, valid)
    if dump_play:
        array = array[:, 2:].astype('int32').tolist()
        n = env.unwrapped.nr_blocks + 1
        array = [array[:n], array[n:]]
        json_str = json.dumps({'array': array, 'moves': moves, 'new_pos': new_pos}) #, indent=True)
        io.mkdir(args.dump_dir)
        dump_file = os.path.join(args.dump_dir, 'blocks_{}_episode_{}.json'.format(env.unwrapped.nr_blocks, episode_id))
        with open(dump_file, 'w') as f:
            f.write(json_str)
    if not gt_steps:
        return succ, score, traj
    else:
        return succ, score, traj, groundtruth_steps


def main():
    logger.info(format_args(args))

    model = Model()
    if args.use_gpu:
        model.cuda()
    optimizer = get_optimizer(args.optimizer, model, args.lr)
    if args.accum_grad > 1:
        optimizer = AccumGrad(optimizer, args.accum_grad)
    trainer = TrainerEnv(model, optimizer)
    if args.load_checkpoint is not None:
        trainer.load_checkpoint(args.load_checkpoint)
    train(trainer)
    save_checkpoint(trainer, 'last')
    test(model)

    if args.visualize_times > 0:
        for i in range(args.visualize_times):
            visualize(model)


if __name__ == '__main__':
    main()
