import signal
import argparse
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from model_statistics import ModelStatistics


class Finish:
    def __init__(self):
        self.__finished = False
        
    def __call__(self, signum, stack):
        if self.__finished:
            exit()
        self.__finished = True
    
    def finished(self):
        return self.__finished
    

def plot_loop(names, paths, title, save=None):
    finish = Finish()
    signal.signal(signal.SIGINT, finish)
    
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
        
    while not finish.finished():    
        stats = {name: ModelStatistics.load_from_file(path) for name, path in zip(names, paths)}
        
        fig, (plt_loss, plt_train_acc, plt_val_acc) = plt.subplots(3, 1)
        fig.set_size_inches(18, 20)
        fig.suptitle(title, fontsize=20)
        fig.tight_layout()
        
        plt_loss.set_ylabel('Loss (local)')
        plt_loss.set_yscale('log')
        plt_loss.set_xlabel('Epoch')
        
        plt_train_acc.set_ylabel('Train Accuracy, %')
        plt_train_acc.set_xlabel('Epoch')

        plt_val_acc.set_ylabel('Validation Accuracy, %')
        plt_val_acc.set_xlabel('Epoch')
    
        for label, stat in stats.items():
            loss = stat.crop('train_loss')
            train_acc = stat.crop('train_precision')
            val_acc = stat.crop('val_precision')
                
            plt_loss.plot(range(len(loss)), loss, label=label)
            plt_train_acc.plot(range(len(train_acc)), train_acc, label=label)
            plt_val_acc.plot(range(len(val_acc)), val_acc, label=label)
    
        plt_loss.legend()
        plt_val_acc.legend()
        plt_train_acc.legend()

        plt.close(fig)
        clear_output(wait = True)
        display(fig)
    
        if save is not None:
            fig.savefig(save)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Online model statistics display')
    parser.add_argument('--names', '-n', nargs='+')
    parser.add_argument('--paths', '-p', nargs='+')
    parser.add_argument('--title', '-t', required=True)
    parser.add_argument('--save', '-s', nargs='?', default=None)
    args = parser.parse_args()
    
    plot_loop(args.names, args.paths, args.title, args.save)
