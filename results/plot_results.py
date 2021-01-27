import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

if __name__ == '__main__':

    path = os.path.join(os.getcwd(), 'data')

    for model_type in ['LSTM', 'NTM']:  # 'NTM',
        plt.figure(dpi=300)

        version = list(range(0, 10)) if model_type == 'LSTM' else list(range(10, 20))
        version_startswith = tuple(['run-version_' + str(i) + '-' for i in version])

        cost_dict = {}
        cost_list = []
        for metric_type in ['train_cost', 'valid_cost']:
            for file in os.listdir(path):
                if file.startswith(version_startswith) and file.endswith(metric_type + '.csv'):
                    # get data
                    data = pd.read_csv(os.path.join(os.getcwd(), 'data', file))

                    step = np.asarray(data['Step'])
                    cost = np.asarray(data['Value'])

                    cost_list.append(cost)

            cost_dict[metric_type] = np.asarray(cost_list)

            # plotting
            plt.errorbar(step, cost_dict[metric_type].mean(axis=0), cost_dict[metric_type].std(axis=0), fmt='-o',
                         capsize=3, label=metric_type)
        plt.gca().set_ylim(bottom=0)
        plt.legend()
        plt.xlabel('Number of batches')
        plt.ylabel('Cost')
        plt.title(model_type + ' cost during training')

        file_format = 'png'
        file_name = model_type.lower() + '_results.' + file_format
        plt.savefig(os.path.join(os.getcwd(), file_name), format=file_format)
        plt.show()
