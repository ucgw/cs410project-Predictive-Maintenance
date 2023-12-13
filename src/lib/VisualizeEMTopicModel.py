import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
from matplotlib.font_manager import FontProperties


def get_top_topic_words_all(ordered_tokens: list[str], log_P: np.ndarray, N: int) -> list[list[str]]:
    """
    Get the top N words for each topic based on their probabilities.

    Parameters:
    - ordered_tokens (list[str]): The list of ordered words
    - log_P (np.ndarray): The matrix of log probabilities for each topic and word
    - N (int): The number of top words to retrieve for each topic

    Returns:
    - list[list[str]]: A list containing the top N words for each topic.
    """
    top_indices = np.argsort(log_P, axis=1)[:, ::-1][:, :N]
    return [[ordered_tokens[x] for x in top_indices_row] for top_indices_row in top_indices]


def get_top_topic_probability(log_pi: np.ndarray) -> tuple[int, float]:
    """
    Get the index and probability of the topic with the highest probability.

    Parameters:
    - log_pi (np.ndarray): The matrix of log probabilities for each topic.

    Returns:
    - tuple[int, float]: The index and probability of the top topic.
    """

    e_log_pi = np.exp(log_pi)

    max_val = np.max(e_log_pi)
    max_idx = np.argmax(e_log_pi)

    return max_idx, max_val


def get_top_topic_words_idx(topic_idx: int, ordered_tokens: list[str], log_P: np.ndarray, N: int) -> list[str]:
    """
    Get the top N words for a specific topic based on their probabilities.

    Parameters:
    - topic_idx (int): The index of the topic.
    - ordered_tokens (list[str]): The list of ordered words.
    - log_P (np.ndarray): The matrix of log probabilities for each topic and word.
    - N (int): The number of top words to retrieve for the specified topic.

    Returns:
    - list[str]: A list containing the top N words for the specified topic.
    """
    return [ordered_tokens[x] for x in np.argsort(log_P[topic_idx])[::-1][:N]]


def viz_topic_freqs(ordered_tokens: list[str], log_pi: np.ndarray, log_P: np.ndarray, topics: int, topic_idx: int,
                    N: int):
    """
    Visualize topic frequencies and top words for each topic.

    Parameters:
    - ordered_tokens (list[str]): The list of ordered words.
    - log_pi (np.ndarray): The matrix of log probabilities for each topic.
    - log_P (np.ndarray): The matrix of log probabilities for each topic and word.
    - topics (int): The number of topics.
    - topic_idx (int): The index of the topic to visualize.
    - N (int): The number of top words to display for each topic.
    """
    plt.rcParams["figure.autolayout"] = True

    plt.rcParams["figure.figsize"] = [7.50, 3.50]

    npexp = np.exp(log_pi)
    maxval = max(npexp)
    maxvalidx = np.where(npexp == maxval)[0][0]
    colors = [ 'gray' for logp in npexp ]
    colors[maxvalidx] = 'red'

    fig, (ax, ax_table) = plt.subplots(nrows=2, gridspec_kw=dict(height_ratios=[3, 1]))
    ax.bar(np.arange(topics), np.exp(log_pi).reshape(-1), color=colors)

    ax.set_title(f'Topic Frequencies')
    ax.set_xlabel(f'Topic Number')
    ax.set_ylabel(f'Topic Density')

    top_words = get_top_topic_words_all(ordered_tokens, log_P, N)

    print(f"Top words Topic[{topic_idx}]: {top_words[topic_idx]}")

    col_labels = ['1st Word', '2nd Word', '3rd Word']
    for i in range(4, N+1):
        col_labels.append(f'{i}th Word')

    row_labels = []
    for idx in range(log_P.shape[0]):
        row_labels.append(f'Topic {idx}')

    ax_table = plt.table(cellText=top_words, rowLabels=row_labels, colLabels=col_labels, cellLoc='center',
                         loc='lower center')

    for (row, col), cell in ax_table.get_celld().items():
        if row == 0 or col == -1:
            cell.set_text_props(fontproperties=FontProperties(weight='bold'))
        else:
            cell.set_text_props(fontproperties=FontProperties(size='large'))

    ax_table.axes.axis('tight')
    ax_table.axes.axis('off')

    plt.subplots_adjust(bottom=0.4)

    fig.patch.set_visible(False)

    plt.show()
