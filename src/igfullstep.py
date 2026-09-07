import os
import argparse
import numpy as np
import torch
import json
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from captum.attr import IntegratedGradients

from model1 import (
    InMemoryNPZDataset,
    collate_fn,
    BaseModel
)


CLASS_NAMES = [
    "Membrane",
    "Cytoplasm",
    "Nucleus",
    "Extracellular",
    "Cell_membrane",
    "Mitochondrion",
    "Plastid",
    "Endoplasmic_reticulum",
    "Lysosome_Vacuole",
    "Golgi_apparatus",
    "Peroxisome"
]


# ======================
# args
# ======================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--npz_embeddings",required=True)
    p.add_argument("--label_csv",required=True)
    p.add_argument("--model_pth",required=True)
    p.add_argument("--output_dir",required=True)

    p.add_argument("--index_dir",default=None)

    p.add_argument("--mode",
                   default="both",
                   choices=["train","val","both"])

    p.add_argument("--n_steps",
                   type=int,
                   default=50)

    p.add_argument("--topk",
                   type=int,
                   default=1000)


    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available()
        else "cpu"
    )


    return p.parse_args()



# ======================
# utils
# ======================

def zscore(x):
    std = x.std()

    if torch.isnan(std) or std < 1e-8:
        return torch.zeros_like(x)

    return (x - x.mean()) / (std + 1e-8)






def plot_heat(x,title,path):

    plt.figure(figsize=(12,2))

    plt.imshow(
        x[np.newaxis,:],
        aspect="auto"
    )

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()



# ======================
# model
# ======================

def build_model(args):

    model = BaseModel(
        embed_dim=1280,
        num_classes=11,
        dropout_rate=0.15
    ).to(args.device)


    model.load_state_dict(
        torch.load(
            args.model_pth,
            map_location=args.device
        )
    )


    model.eval()


    # remove dropout randomness
    for m in model.modules():

        if isinstance(m,torch.nn.Dropout):
            m.p=0


    return model



# ======================
# forward
# ======================


def forward(model,x,lengths,mask):

    logits,_ = model(
        x,
        lengths,
        mask
    )

    return logits



# ======================
# grad norm
# ======================


def grad_norm(
        model,
        seq,
        length,
        mask,
        target
):


    seq = (
        seq.unsqueeze(0)
        .detach()
        .clone()
        .requires_grad_(True)
    )


    logits,_ = model(
        seq,
        length.unsqueeze(0),
        mask.unsqueeze(0)
    )


    logits[0,target].backward()


    g = seq.grad.abs().mean(dim=-1)


    return g.squeeze(0).detach()



# ======================
# main pipeline
# ======================


def run(args,dataset,split):


    outdir=os.path.join(
        args.output_dir,
        split
    )


    os.makedirs(outdir,exist_ok=True)


    model=build_model(args)



    ig=IntegratedGradients(
        lambda x,l,m:
        forward(model,x,l,m)
    )



    loader=DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )



    for batch in loader:


        x,lengths,mask,y,names,_=batch


        x=x.to(args.device)
        lengths=lengths.to(args.device)
        mask=mask.to(args.device)


        seq=x[0]
        L=lengths[0]
        m=mask[0]

        name=names[0]



        with torch.no_grad():


            logits,attn=model(
                seq.unsqueeze(0),
                L.unsqueeze(0),
                m.unsqueeze(0)
            )


            probs=torch.sigmoid(
                logits[0]
            )



            pred_classes=torch.where(
                probs>0.5
            )[0]


            true_classes=torch.where(
                y[0]>0.5
            )[0]



        # 只解释正确类别

        classes=list(
            set(
                pred_classes.cpu().numpy()
            )
            &
            set(
                true_classes.cpu().numpy()
            )
        )


        if len(classes)==0:
            continue



        # attention

        attn=attn[0]


        attn_score=attn.mean(dim=-1)


        # remove CLS

        attn_score=attn_score[1:]




        for cls in classes:



            cls_name=CLASS_NAMES[cls]



            ig_attr=ig.attribute(

                seq.unsqueeze(0),

                baselines=
                torch.zeros_like(
                    seq.unsqueeze(0)
                ),


                additional_forward_args=(

                    L.unsqueeze(0),
                    m.unsqueeze(0)

                ),


                target=int(cls),

                n_steps=args.n_steps

            ).squeeze(0)



            grad=grad_norm(

                model,
                seq,
                L,
                m,
                int(cls)

            )



            ig_score=(
                ig_attr
                .abs()
                .sum(dim=-1)
            )



            fusion=(

                0.45*zscore(ig_score)

                +

                0.35*zscore(attn_score)

                +

                0.20*zscore(grad)

            )



            fusion=fusion-fusion.min()

            fusion = fusion / (fusion.max() + 1e-8)



            fusion_np=fusion.cpu().numpy()



            top10=np.argsort(
                fusion_np
            )[::-1][:10]



            save_dir=os.path.join(

                outdir,

                "correct_class_residue",

                cls_name

            )


            os.makedirs(
                save_dir,
                exist_ok=True
            )



            np.savez_compressed(

                os.path.join(
                    save_dir,
                    f"{name}.npz"
                ),


                name=name,

                class_name=cls_name,

                class_id=int(cls),

                probability=float(
                    probs[cls]
                    .item()
                ),


                ig_score=
                ig_score.cpu().numpy(),


                attention=
                attn_score.cpu().numpy(),


                grad=
                grad.cpu().numpy(),


                fusion=fusion_np,


                top10_idx=top10

            )



            plot_heat(

                fusion_np,

                f"{name}-{cls_name}",


                os.path.join(
                    save_dir,
                    f"{name}.png"
                )

            )



    print(
        split,
        "done"
    )



# ======================
# main
# ======================


def main():

    args=parse_args()


    dataset=InMemoryNPZDataset(
        args.npz_embeddings,
        args.label_csv,
        max_samples=30000
    )



    if args.index_dir:


        train_idx=np.load(
            os.path.join(
                args.index_dir,
                "train_indices.npy"
            )
        )


        val_idx=np.load(
            os.path.join(
                args.index_dir,
                "val_indices.npy"
            )
        )


        train_ds=Subset(
            dataset,
            train_idx
        )


        val_ds=Subset(
            dataset,
            val_idx
        )


    else:

        n=len(dataset)

        train_ds=Subset(
            dataset,
            range(int(n*0.8))
        )


        val_ds=Subset(
            dataset,
            range(int(n*0.8),n)
        )



    if args.mode in ["train","both"]:

        run(
            args,
            train_ds,
            "train"
        )



    if args.mode in ["val","both"]:

        run(
            args,
            val_ds,
            "val"
        )



if __name__=="__main__":

    main()