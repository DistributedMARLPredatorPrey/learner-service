import pandas as pd
import os


def setup_pred_prey_losses_files(root_project_path):
    """
    Setups the files for storing Critic network losses
    :param root_project_path: Root path of the project
    :return:
    """
    base_path: str = os.path.join(root_project_path, "src", "main", "resources", "loss")
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        os.remove(item_path)


def save_pred_prey_losses(root_project_path, predator_losses, prey_losses):
    """
    Save loss of Predator and Prey Critic network to file
    :param root_project_path: Root path of the project
    :param predator_losses: Losses of Predator's Critic
    :param prey_losses: Losses of Prey's Critic
    :return:
    """
    base_path: str = os.path.join(root_project_path, "src", "main", "resources", "loss")
    pred_losses_path: str = os.path.join(base_path, "df_predator_losses.csv")
    prey_losses_path: str = os.path.join(base_path, "df_prey_losses.csv")

    predator_losses = [loss.numpy() for loss in predator_losses]
    prey_losses = [loss.numpy() for loss in prey_losses]

    df_pred_losses = pd.DataFrame({"pred_loss": predator_losses})
    df_prey_losses = pd.DataFrame({"prey_loss": prey_losses})

    if os.path.exists(pred_losses_path) and os.path.exists(prey_losses_path):
        df_ex_pred_losses = pd.read_csv(pred_losses_path, index_col=0)
        df_ex_prey_losses = pd.read_csv(prey_losses_path, index_col=0)
        pd.concat([df_ex_pred_losses, df_pred_losses], ignore_index=True).to_csv(
            pred_losses_path
        )
        pd.concat([df_ex_prey_losses, df_prey_losses], ignore_index=True).to_csv(
            prey_losses_path
        )
    else:
        df_pred_losses.to_csv(pred_losses_path)
        df_prey_losses.to_csv(prey_losses_path)
