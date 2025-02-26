import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image

model_color = 'green'
def render_animation(skeleton, poses_generator, algos, t_hist, fix_0=True, azim=0.0, output=None, mode='pred', size=2,
                     ncol=5,
                     bitrate=3000, fix_index=None):
    if mode == 'switch':
        fix_0 = False
    if fix_index is not None:
        fix_list = [
            [1, 2, 3],  #
            [4, 5, 6],
            [7, 8, 9, 10],
            [11, 12, 13],
            [14, 15, 16],
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        ]
        fix_i = fix_list[fix_index]
        fix_col = 'darkblue'
    else:
        fix_i = None
    all_poses = next(poses_generator)
    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    t_total = next(iter(all_poses.values())).shape[0]
    poses = dict(filter(lambda x: x[0] in {'preds', 'trues'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                        all_poses.items()))
 
    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):

        if index < ncol:
            ax = fig.add_subplot(1, ncol, index + 1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
   
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    fig.tight_layout(h_pad=15, w_pad=15)
    fig.subplots_adjust(wspace=-0.1, hspace=0.5)
    poses = list(poses.values())

    anim = None
    initialized = False
    animating = True
    find = 0
    hist_lcol, hist_mcol, hist_rcol = 'red', 'red', 'red'
    pred_lcol, pred_mcol, pred_rcol = 'red', 'red', 'red'
    tran_lcol, tran_mcol, tran_rcol = 'red', 'red', 'red'

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized
        if mode == 'switch':
            if i < t_hist:
                lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
            elif i > 75:
                lcol, mcol, rcol = tran_lcol, pred_mcol, tran_rcol
            else:
                lcol, mcol, rcol = pred_lcol, tran_mcol, pred_rcol
        else:
            if i < t_hist:
                lcol, mcol, rcol = hist_lcol, hist_mcol, hist_rcol
            else:
                lcol, mcol, rcol = pred_lcol, pred_mcol, pred_rcol

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            if fix_0 and n % ncol == 0 and i >= t_hist:
                continue
            trajectories[n] = poses[n][:, 0, [0, 1, 2]]
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius / 2 + trajectories[n][i, 2], radius / 2 + trajectories[n][i, 2]])
        if not initialized:
            for j, j_parent in enumerate(parents):

                if j_parent == -1:
                    continue

                if j in skeleton.joints_right():
                    col = rcol
                elif j in skeleton.joints_left():
                    col = lcol
                else:
                    col = mcol
                # if j == 0:
                #     col = 'blue'

                if fix_i is not None and j in fix_i:
                    col = fix_col

                for n, ax in enumerate(ax_3d):
                    # if n in [0, 6]:
                    #     continue
                    if n in [5, 6, 7, 8, 9]:
                        continue
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, linewidth=3.0))
                    pos = poses[n+5][i]
                    lines_3d[n+5].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col, linewidth=3.0))
            initialized = True
        else:
            # print(1)
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                if j in skeleton.joints_right():
                    col = rcol
                elif j in skeleton.joints_left():
                    col = lcol
                else:
                    col = mcol

                if fix_i is not None and j in fix_i:
                    col = fix_col

                for n, ax in enumerate(ax_3d):
                    # if n in [0, 6]:
                    #     continue
                    if n in [5, 6, 7, 8, 9]:
                        continue

                    # if fix_0 and n == 0 and i >= t_hist:
                    #     continue
                    # if fix_0 and n % ncol == 0 and i >= t_hist:
                    #     continue

                    pos = poses[n][i]
                    x_array = np.array([pos[j, 0], pos[j_parent, 0]])
                    y_array = np.array([pos[j, 1], pos[j_parent, 1]])
                    z_array = np.array([pos[j, 2], pos[j_parent, 2]])
                    lines_3d[n][j - 1][0].set_data_3d(x_array, y_array, z_array)
                    lines_3d[n][j - 1][0].set_color(col)

                    pos = poses[n+5][i]
                    x_array = np.array([pos[j, 0], pos[j_parent, 0]])
                    y_array = np.array([pos[j, 1], pos[j_parent, 1]])
                    z_array = np.array([pos[j, 2], pos[j_parent, 2]])
                    lines_3d[n+5][j - 1][0].set_data_3d(x_array, y_array, z_array)
                    lines_3d[n+5][j - 1][0].set_color(model_color)

    def show_animation():
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                            all_poses.items()))
        if x[0] in {'gt', 'context'}:
            for ax, title in zip(ax_3d, poses.keys()):
                ax.set_title(title, y=1.0, fontsize=12)
        if mode == 'switch':
            if x[0] in {algo + '_0'}:
                for ax, title in zip(ax_3d, poses.keys()):
                    ax.set_title('target', y=1.0, fontsize=12)

        poses = list(poses.values())
    # def save_figs():
    #     nonlocal algo, find
    #     old_algo = algo
    #     for algo in algos:
    #         reload_poses()
    #         update_video(t_total - 1)
    #         fig.savefig('out/%d_%s.png' % (find, algo), dpi=400, transparent=True)
    #     algo = old_algo
    #     find += 1

    def save_figs():
        nonlocal algo, find
        old_algo = algo
        os.makedirs('out_svg', exist_ok=True)
        suffix = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')[:-3]
        os.makedirs('out_svg_' + suffix, exist_ok=True)
        for algo in algos:
            reload_poses()
            for i in range(0, t_total + 1, 10):
                if i == 0:
                    update_video(0)
                else:
                    update_video(i - 1)
                fig.savefig('out_svg_' + suffix + '/%d_%s_%d.svg' % (find, algo, i), transparent=True)

                # img = Image.open('out_svg_' + suffix + '/%d_%s_%d.svg' % (find, algo, i))
                # box = (100, 100, 400, 400)
                # cropped_img = img.crop(box)
                # cropped_img.save('cropped_example.jpg')

        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim

        if event.key == 'd':
            all_poses = next(poses_generator)
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        fps = 50
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='pillow')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    fig.canvas.mpl_connect('key_press_event', on_key)

    save()
    show_animation()
    plt.show()
    plt.close()

    # save_figs()

def pose_generator(trues, pred):
    """
    stack k rows examples in one gif

    The logic of 'draw_order_indicator' is to cheat the render_animation(),
    because this render function only identify the first two as context and gt, which is a bit tricky to modify.
    """
    while True:
        # gt = trues[0:125, :, :].copy()  # 125*17*3
        # gt[:, :1, :] = 0
        # trues= trues[0:750,:,:]
        # trues = trues1.reshape(375, 17, 3)
        # pred = pred.reshape(750, 17, 3)
        poses = {}
        # trues= trues[0:1,:,:,:]
        gt1 = trues[0:125,:,:]
        gt1[:, :1, :] = 0
        gt2 = pred[0:125,:,:]
        gt2[:, :1, :] = 0
        poses['trues'] = gt1

        # poses['gt'] = gt
        for i in range(4):
            x = trues[125 * (i+1):125 * (i + 2), :, :]#125*17*3
            x[:, :1, :] = 0
            poses[f'HumanMAC_{i}'] = x
        poses['preds'] = gt2
        for i in range(4):
            x = pred[125 * (i+1):125 * (i + 2), :, :]#125*17*3
            x[:, :1, :] = 0
            poses[f'HumanMAC_{i+6}'] = x

        yield poses

class Skeleton:
    def __init__(self, parents, joints_left, joints_right):
        assert len(joints_left) == len(joints_right)

        self._parents = np.array(parents)
        self._joints_left = joints_left
        self._joints_right = joints_right
        self._compute_metadata()

    def num_joints(self):
        return len(self._parents)

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove):
        """
        Remove the joints specified in 'joints_to_remove'.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        for i in range(len(self._parents)):
            while self._parents[i] in joints_to_remove:
                self._parents[i] = self._parents[self._parents[i]]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        if self._joints_left is not None:
            new_joints_left = []
            for joint in self._joints_left:
                if joint in valid_joints:
                    new_joints_left.append(joint - index_offsets[joint])
            self._joints_left = new_joints_left
        if self._joints_right is not None:
            new_joints_right = []
            for joint in self._joints_right:
                if joint in valid_joints:
                    new_joints_right.append(joint - index_offsets[joint])
            self._joints_right = new_joints_right

        self._compute_metadata()

        return valid_joints

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

import argparse

if __name__ == '__main__':
    skeleton = Skeleton(parents=[-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
                                 16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
                        joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
                        joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])
    removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
    skeleton.remove_joints(removed_joints)
    skeleton._parents[11] = 8
    skeleton._parents[14] = 8

    print('process')

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default=['Walking_all'], type=str, nargs='+')
    parser.add_argument('-models', default=['IDOL'], type=str, nargs='+')
    parser.add_argument('-color', default='green', type=str)
    parser.add_argument('-seed', default=[2024], type=int, nargs='+')
    parser.add_argument('-pred_len', default=[125], type=int, nargs='+')

    args = parser.parse_args()

    model_color = args.color

    draw_path = './draw_results'
    save_path = './save_draw_results'

    for dataset in args.dataset:
        for model in args.models:
            for l in args.pred_len:
                for seed in args.seed:
                    trues = np.load(f'{draw_path}/Human/{dataset}/{model}/{l}_{seed}/trues.npy')
                    preds = np.load(f'{draw_path}/Human/{dataset}/{model}/{l}_{seed}/preds.npy')
                    n = int((trues.shape[0])/625)
                    trues = trues[:n*625, :]
                    preds = preds[:n*625, :]
                    trues = trues.reshape(-1, 17, 3)
                    preds = preds.reshape(-1, 17, 3)
                    for i in tqdm(range(n)):
                        tr = trues[i*625:(i+1)*625,:,:]
                        pr = preds[i*625:(i+1)*625,:,:]
                        pose_gen = pose_generator(tr,pr)
                        render_animation(skeleton, pose_gen, ['HumanMAC'], 25, ncol=5,
                                         output=os.path.join(f'{save_path}/{dataset}/{model}/{l}_{seed}', f'{i}.gif'))
