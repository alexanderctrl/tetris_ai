from pathlib import Path

import matplotlib.pyplot as plt

plt.ion()


def plot_training_progress(
    scores: list[int],
    mean_scores: list[float],
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
    save_and_close : bool, optional
        Boolean flag that indicates whether to save and close the plot.
    file_name : str, optional
        Name of the file to save the plot to. Default: ``"training_progress.pdf"``.
    """
    plt.clf()
    plt.style.use("ggplot")
    plt.title("Training Progess")
    plt.xlabel("Episode")
    plt.ylabel("Score")

    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")

    plt.text(len(scores) - 1, scores[-1], f"{scores[-1]}", ha="left")
    plt.text(len(mean_scores) - 1, mean_scores[-1], f"{mean_scores[-1]:.2}", ha="left")

    plt.ylim(ymin=0)
    plt.legend(loc="upper left")
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
