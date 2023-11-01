from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import torch.nn.functional as F

from model import GCN, LinearGCN, GAT, MLP, Model

def train_epoch(model, model_name, data, optimizer, class_weight=None):
    out = forward(model, model_name, data)
    loss = F.nll_loss(F.log_softmax(out[data.train_mask], dim=1), data.y[data.train_mask], weight=class_weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def eval_epoch(model, model_name, data, test=False, class_weight=None):
    out = forward(model, model_name, data, eval=True)
    if test:
        mask = data.test_mask
    else:
        mask = data.val_mask
    loss = F.nll_loss(F.log_softmax(out[mask], dim=1), data.y[mask], weight=class_weight)
    y_pred = torch.argmax(out[mask], dim=1).cpu().numpy()
    y_prob = torch.softmax(out[mask], dim=1).cpu().numpy()
    y_true = data.y[mask].cpu().numpy()
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    return loss.item(), acc, bacc, y_pred, y_prob, y_true

def forward(model, model_name, data, eval=False):
    if model_name == "mlp":
        if eval:
            model.eval()
            with torch.no_grad():
                out = model(data.x)
        else:
            model.train()
            out = model(data.x)
    else:
        if eval:
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
        else:
            model.train()
            out = model(data.x, data.edge_index)
    return out


def torch_fit(model, model_name, data, optimizer, epochs, class_weight=None):
    for e in range(1, epochs + 1):
        train_loss = train_epoch(model, model_name, data, optimizer, class_weight=class_weight)
        val_loss, val_acc, val_bacc, _, _, _ = eval_epoch(model, model_name, data, class_weight=class_weight)
        print(f"Epoch: {e} Train Loss: {train_loss} Val Loss: {val_loss} Val Acc: {val_acc} Val Bacc: {val_bacc}")


def create_model(model_name, device, hyper_param):
    if model_name == "mlp":
        model = MLP(hyper_param["mlp_dim_list"], dropout_list=hyper_param["mlp_dr_list"]).to(device)
    elif model_name == "lingcn":
        model = LinearGCN(hyper_param["gnn_dim_list"]).to(device)
    elif model_name == "gcn":
        encoder = GCN(hyper_param["gnn_dim_list"], dropout_list=hyper_param["gnn_dr_list"])
        mlp = MLP(hyper_param["mlp_dim_list"], dropout_list=hyper_param["mlp_dr_list"])
        model = Model(encoder, mlp).to(device)
    elif model_name == "gat":
        encoder = GAT(hyper_param["gnn_dim_list"], dropout_list=hyper_param["gnn_dr_list"])
        mlp = MLP(hyper_param["mlp_dim_list"], dropout_list=hyper_param["mlp_dr_list"])
        model = Model(encoder, mlp).to(device)
    return model