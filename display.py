import signal
import argparse
import time
import os
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from model_statistics import ModelStatistics


class Finish:
    def __init__(self):
        self.__finished = False

    def __call__(self, signum, stack):
        if self.__finished:
            os.exit()
        self.__finished = True

    def finished(self):
        return self.__finished


def plot_loop(names, paths, title, save=None, param_dev=None):
    finish = Finish()
    signal.signal(signal.SIGINT, finish)

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'

    while not finish.finished():
        stats = {name: ModelStatistics.load_from_file(path) for name, path in zip(names, paths)}
        param_dev_stats = ModelStatistics.load_from_file(param_dev) if param_dev else None

        if param_dev_stats:
            fig, (plt_loss, plt_param_dev, plt_val_acc) = plt.subplots(3, 1)
            fig.set_size_inches(18, 20)
        else:
            plt_param_dev = None
            fig, (plt_loss, plt_val_acc) = plt.subplots(2, 1)
            fig.set_size_inches(18, 20 * 0.7)
        fig.suptitle(title, fontsize=20)

        plt_loss.set_ylabel('Loss (local)')
        plt_loss.set_yscale('log')
        plt_loss.set_xlabel('Epoch')

        if param_dev_stats:
            plt_param_dev.set_ylabel('Parameter deviation (coef. of variation)')
            plt_param_dev.set_xlabel('Epoch')
            plt_param_dev.set_yscale('log')

        plt_val_acc.set_ylabel('Validation Accuracy, %')
        plt_val_acc.set_xlabel('Epoch')

        for label, stat in stats.items():
            loss = stat.crop('train_loss')
            val_acc = stat.crop('val_precision')
            
            fmt = {}
            if label.lower().find('consensus') != -1:
                fmt['linestyle'] = 'dashed'
                fmt['linewidth'] = 1.1
            else:
                fmt['linestyle'] = None
                fmt['linewidth'] = 1.5

            plt_loss.plot(range(len(loss)), loss, label=label, **fmt)
            plt_val_acc.plot(range(len(val_acc)), val_acc, label=label + ' ({})'.format(val_acc[-1]), **fmt)

        if param_dev_stats:
            telemetries_per_epoch = next(iter(param_dev_stats.crop('telemetries_per_epoch')[0].values()))
            deviation = param_dev_stats.crop('coef_of_var')
            plt_param_dev.plot([b / telemetries_per_epoch for b in range(len(deviation))],
                                deviation, label='max')
            try:
                try:
                    cv_pctls = param_dev_stats.crop('abs_coef_of_var_percentiles')
                except:
                    cv_pctls = param_dev_stats.crop('coef_of_var_percentiles')
                grouped_by_pcts = dict()
                for record in cv_pctls:
                    for (pct, val) in record:
                        if pct not in grouped_by_pcts.keys():
                            grouped_by_pcts[pct] = []
                        grouped_by_pcts[pct].append(val)
                for pct, vals in reversed(list(grouped_by_pcts.items())):
                    if pct < 75 or 99 < pct:
                        continue
                    plt_param_dev.plot([b / telemetries_per_epoch for b in range(len(vals))],
                                       vals, label='percentile={}'.format(pct))
            except:
                pass

        plt_loss.legend()
        plt_val_acc.legend()
        if param_dev_stats:
            plt_param_dev.legend()
            
        fig.tight_layout()
        plt.close(fig)
        clear_output(wait=True)
        display(fig)

        if save is not None:
            fig.savefig(save)

        time.sleep(5.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online model statistics display')
    parser.add_argument('--names', '-n', nargs='+')
    parser.add_argument('--paths', '-p', nargs='+')
    parser.add_argument('--title', '-t', required=True)
    parser.add_argument('--save', '-s', nargs='?', default=None)
    parser.add_argument('--param-dev', required=False, default=None,
                        help='model statistics file with parameter deviation (produced by master)')
    args = parser.parse_args()

    plot_loop(args.names, args.paths, args.title, args.save, args.param_dev)
