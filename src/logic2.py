import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)


# =====================================================
# Attention Head (model2)
# =====================================================

class AttentionHead(nn.Module):

    def __init__(
        self,
        hidden_dim,
        n_heads=8,
        dropout_rate=0.15
    ):
        super().__init__()

        self.n_heads = n_heads


        self.dropout = nn.Dropout(
            dropout_rate
        )


        self.preattn_ln = nn.LayerNorm(
            hidden_dim // n_heads
        )


        self.Q = nn.Linear(
            hidden_dim // n_heads,
            n_heads,
            bias=False
        )


        torch.nn.init.normal_(
            self.Q.weight,
            mean=0.0,
            std=1/(hidden_dim//n_heads)
        )



    def forward(
        self,
        x,
        mask,
        importance=None
    ):


        B,L,D = x.shape



        x = x.view(
            B,
            L,
            self.n_heads,
            D//self.n_heads
        )


        x = self.preattn_ln(x)



        attn = (
            x *
            self.Q.weight.view(
                1,
                1,
                self.n_heads,
                D//self.n_heads
            )
        ).sum(-1)



        # -------------------------
        # importance bias
        # 测试时 None
        # -------------------------

        if importance is not None:


            importance = torch.log1p(
                importance
            )


            importance = (
                importance -
                importance.mean(
                    dim=1,
                    keepdim=True
                )
            ) / (
                importance.std(
                    dim=1,
                    keepdim=True
                )
                +1e-8
            )


            importance = importance.unsqueeze(-1)


            attn = attn + 0.5 * importance




        attn = attn.masked_fill(
            ~mask.unsqueeze(-1),
            -1e9
        )


        attn = F.softmax(
            attn,
            dim=1
        )


        attn = self.dropout(
            attn
        )


        x = (
            x *
            attn.unsqueeze(-1)
        ).sum(1)



        x = x.view(
            B,
            -1
        )



        return x, attn.squeeze(-1)





# =====================================================
# BaseModel (model2)
# =====================================================

class BaseModel(nn.Module):

    def __init__(
        self,
        embed_dim=1280,
        num_classes=11,
        dropout_rate=0.15
    ):
        super().__init__()



        self.initial_ln = nn.LayerNorm(
            embed_dim
        )


        self.lin = nn.Linear(
            embed_dim,
            512
        )


        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                embed_dim
            )
        )


        self.dropout = nn.Dropout(
            dropout_rate
        )


        self.attn_head = AttentionHead(
            512,
            n_heads=8,
            dropout_rate=dropout_rate
        )


        self.clf_head = nn.Linear(
            512,
            num_classes
        )




    def forward(
        self,
        x,
        mask,
        importance=None
    ):


        B=x.size(0)



        cls_token = self.cls_token.expand(
            B,
            -1,
            -1
        )


        x=torch.cat(
            [
                cls_token,
                x
            ],
            dim=1
        )



        mask=torch.cat(
            [
                torch.ones(
                    B,
                    1,
                    dtype=torch.bool,
                    device=x.device
                ),
                mask
            ],
            dim=1
        )



        x=self.initial_ln(
            x
        )


        x=self.lin(
            x
        )


        x=F.relu(
            x
        )


        x=self.dropout(
            x
        )



        # 测试没有importance
        if importance is not None:

            importance=torch.cat(
                [
                    torch.zeros(
                        B,
                        1,
                        device=x.device
                    ),
                    importance
                ],
                dim=1
            )



        x_pool,attn=self.attn_head(
            x,
            mask,
            importance
        )


        logits=self.clf_head(
            x_pool
        )



        return logits,attn






# =====================================================
# HPA Dataset
# 无 importance
# =====================================================

class HPADataset(Dataset):

    def __init__(
        self,
        npz_file,
        label_csv
    ):


        self.data_npz=np.load(
            npz_file,
            allow_pickle=True
        )


        self.keys=list(
            self.data_npz.keys()
        )


        self.label_df=pd.read_csv(
            label_csv
        ).set_index(
            "sid"
        )


        self.embeddings=[]
        self.labels=[]
        self.masks=[]



        self.label_cols=[
            "Cell membrane",
            "Cytoplasm",
            "Endoplasmic reticulum",
            "Golgi apparatus",
            "Mitochondrion",
            "Nucleus"
        ]



        for k in self.keys:



            emb=torch.tensor(
                self.data_npz[k],
                dtype=torch.float32
            )


            self.embeddings.append(
                emb
            )


            self.masks.append(
                torch.ones(
                    len(emb),
                    dtype=torch.bool
                )
            )



            sid=str(k)



            label=torch.tensor(
                self.label_df.loc[
                    sid,
                    self.label_cols
                ].values,
                dtype=torch.float32
            )



            self.labels.append(
                label
            )



    def __len__(self):

        return len(self.keys)



    def __getitem__(self,idx):

        return (
            self.embeddings[idx],
            self.masks[idx],
            self.labels[idx],
            self.keys[idx]
        )






# =====================================================
# Collate
# =====================================================

def collate_fn(batch):


    embeddings,masks,labels,names=zip(*batch)



    embeddings=pad_sequence(
        embeddings,
        batch_first=True
    )


    masks=pad_sequence(
        masks,
        batch_first=True
    )


    labels=torch.stack(
        labels,
        dim=0
    )



    return (
        embeddings,
        masks,
        labels,
        names
    )






# =====================================================
# Paths
# =====================================================


MODEL_PTH="/path/to/project/recycle_results7/cycle_2/train_output/recycle_model_weights.pth"


TEST_NPZ="/path/to/project/sequence_embeddings_avg1_hpafull.npz"


LABEL_CSV="/path/to/project/hpa_testset.csv"


OUTPUT_DIR="/path/to/project/recycle_results7/cycle_2/train_output"



os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)





# =====================================================
# Device
# =====================================================

device="cuda" if torch.cuda.is_available() else "cpu"







# =====================================================
# Data
# =====================================================


dataset=HPADataset(
    TEST_NPZ,
    LABEL_CSV
)



loader=DataLoader(
    dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_fn
)







# =====================================================
# Load Model
# =====================================================


model=BaseModel(
    embed_dim=1280,
    num_classes=11,
    dropout_rate=0.15
).to(device)



state=torch.load(
    MODEL_PTH,
    map_location=device
)



model.load_state_dict(
    state
)



model.eval()






# =====================================================
# Inference
# =====================================================


y_true_all=[]
y_pred_all=[]



with torch.no_grad():


    for (
        x,
        mask,
        y,
        names
    ) in loader:



        x=x.to(device)

        mask=mask.to(device)



        logits,attn=model(
            x,
            mask,
            importance=None
        )



        # 11 classes -> HPA 6

        logits=logits[
            :,
            [
                4,
                1,
                7,
                9,
                5,
                2
            ]
        ]



        pred=(
            torch.sigmoid(logits)>0.5
        ).cpu().numpy()



        y_pred_all.append(
            pred
        )


        y_true_all.append(
            y.numpy()
        )







# =====================================================
# Metrics
# =====================================================


y_true_all=np.vstack(
    y_true_all
)


y_pred_all=np.vstack(
    y_pred_all
)




metrics={}



metrics["accuracy"]=accuracy_score(
    y_true_all,
    y_pred_all
)


metrics["precision"]=precision_score(
    y_true_all,
    y_pred_all,
    average="macro",
    zero_division=0
)


metrics["recall"]=recall_score(
    y_true_all,
    y_pred_all,
    average="macro",
    zero_division=0
)


metrics["macro_f1"]=f1_score(
    y_true_all,
    y_pred_all,
    average="macro",
    zero_division=0
)


metrics["micro_f1"]=f1_score(
    y_true_all.flatten(),
    y_pred_all.flatten(),
    zero_division=0
)


metrics["total_mcc"]=matthews_corrcoef(
    y_true_all.flatten(),
    y_pred_all.flatten()
)





for i in range(6):


    metrics[f"class_{i}_acc"]=accuracy_score(
        y_true_all[:,i],
        y_pred_all[:,i]
    )


    metrics[f"class_{i}_f1"]=f1_score(
        y_true_all[:,i],
        y_pred_all[:,i],
        zero_division=0
    )


    metrics[f"class_{i}_mcc"]=matthews_corrcoef(
        y_true_all[:,i],
        y_pred_all[:,i]
    )







# =====================================================
# Save
# =====================================================


out_csv=os.path.join(
    OUTPUT_DIR,
    "hpa_test_metrics_model2.csv"
)



pd.DataFrame(
    [metrics]
).to_csv(
    out_csv,
    index=False
)



print("==============================")

for k,v in metrics.items():

    print(
        f"{k}: {v:.4f}"
    )


print("==============================")


print(
    "Saved ->",
    out_csv
)