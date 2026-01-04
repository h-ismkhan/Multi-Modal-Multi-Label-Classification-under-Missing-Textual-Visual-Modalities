"""Federated Cross-Modal Simulation: Client App"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from farm.task import Net, load_data
from farm.task import test as test_fn
from farm.task import train_sim, train_tsk

# Flower ClientApp
app = ClientApp()
current_round_num = 0

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    global current_round_num
    current_round_num += 1
    
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the model and initialize it with the received weights
    model = Net(device=device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)
    
    # Load the data with missing config from server
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    missing_config = msg.content["config"]["missing-config"]
    num_r_for_sim_train: int = context.run_config["num-rounds-for-sim-train"]
    
    trainloader, _ = load_data(partition_id, num_partitions, missing_config)
    
    # Get training hyperparameters
    local_epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]

    num_rounds: int = context.run_config["num-server-rounds"]

    epoch_rmse = {'image': [], 'text': []}
    train_loss = float('inf')
    if current_round_num <= num_r_for_sim_train:
        print(f"We are in round : {current_round_num}, and will train sim.")
        train_loss, epoch_rmse = train_sim(
            model,
            trainloader,
            local_epochs,
            lr,
            device,
        )
    else:
        print(f"We are in round : {current_round_num}, and will train the task.")
        train_loss = train_tsk(
            model,
            trainloader,
            local_epochs,
            lr,
            device,
        )

    if current_round_num == num_rounds:
        current_round_num = 0
    # Construct and return reply Message
    # Send ONLY this round's RMSE (list of local_epochs values)
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        # Send THIS round's RMSE only (size = local_epochs)
        "image_rmse": epoch_rmse['image'],
        "text_rmse": epoch_rmse['text'],
        "client_size": len(trainloader.dataset),  # For weighted averaging
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    
    # Get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the model and initialize it with the received weights
    model = Net(device=device)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)
    
    # Load the data with missing config from server
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    missing_config = msg.content["config"]["missing-config"]
    
    _, valloader = load_data(partition_id, num_partitions, missing_config)
    
    # Call the evaluation function
    eval_f1_micro, eval_f1_macro = test_fn(
        model,
        valloader,
        device,
    )
    
    # Construct and return reply Message
    metrics = {
        "eval_f1_micro": eval_f1_micro,
        "eval_f1_macro": eval_f1_macro,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)