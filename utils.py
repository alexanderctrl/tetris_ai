from pathlib import Path

import matplotlib.cm as cmx
import matplotlib.pyplot as plt

plt.ion()


def plot_training_progress(
    scores: list[int],
    mean_scores: list[float],
    discounted_returns: list[float],
    mean_discounted_returns: list[float],
    save_and_close: bool = False,
    file_name: str = "training_progress.pdf",
) -> None:
    """
    Plot the achieved scores and mean scores over the training episodes in
    real-time during training.

    Parameters
    ----------
    scores : list[int]
        The scores achieved in each training episode.
    mean_scores : list[float]
        The running mean scores up to each episode.
    discounted_returns : list[float]
        The discounted returns achieved in each training episode.
    mean_discounted_returns : list[float]
        The running mean discounted returns up to each episode.
    save_and_close : bool, optional
        Boolean flag that indicates whether to save and close the plot.
    file_name : str, optional
        Name of the file to save the plot to. Default: ``"training_progress.pdf"``.
    """
    plt.clf()
    plt.title("Training Progess")
    plt.xlabel("Episode")

    ax1 = plt.gca()
    cmap_left = cmx.get_cmap("Blues")
    ax1.set_ylabel("Score")
    ax1.set_ylim(ymin=0)
    ax1.plot(scores, label="Score", color=cmap_left(0.95))
    ax1.plot(mean_scores, label="Mean Score", color=cmap_left(0.75))
    ax1.text(len(scores) - 1, scores[-1], f"{scores[-1]}", ha="left")
    ax1.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2}", ha="left")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    cmap_right = cmx.get_cmap("Reds")
    ax2.set_ylabel("Discounted Return")
    ax2.plot(discounted_returns, label="Discounted Return", color=cmap_right(0.95))
    ax2.plot(
        mean_discounted_returns, label="Mean Discounted Return", color=cmap_right(0.75)
    )
    ax2.text(
        len(discounted_returns) - 1,
        discounted_returns[-1],
        f"{discounted_returns[-1]:.2}",
        ha="left",
    )
    ax2.text(
        len(mean_discounted_returns) - 1,
        mean_discounted_returns[-1],
        f"{mean_discounted_returns[-1]:.2}",
        ha="left",
    )
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.01)

    if save_and_close:
        images_folder = Path("images")
        images_folder.mkdir(parents=True, exist_ok=True)
        file_path = images_folder / file_name

        plt.ioff()
        plt.savefig(file_path)
        plt.close()
