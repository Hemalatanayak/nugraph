"""NuGraph3 filter decoder"""
from typing import Any
import tempfile
import matplotlib.pyplot as plt
import seaborn as sn
import torch
from torch import nn
import torchmetrics as tm
from torch_geometric.data import Batch
from pytorch_lightning.loggers import Logger
from ....util import ConfusionMatrixLogger
from ..types import Data

from ....util.DANNLoss import ReverseLayerF
from ....util.EmbeddingPlotter import CombinedEmbeddingPlot


class FilterDecoder(nn.Module):
    """
    NuGraph3 filter decoder module

    Convolve hit node embedding down to a single node score to identify and
    filter out graph nodes that are not part of the primary physics
    interaction.

    Args:
        hit_features: Number of hit node features
    """
    def __init__(self, hit_features: int, warmup_epochs: int = 0, da_loss_fnc_name: str = None):
        super().__init__()

        self.warmup_epochs = warmup_epochs
        self.da_loss_fnc_name = da_loss_fnc_name
        self.use_domain_adaptation = False 
        self.assume_target_label_known = True
        self.domain_adaptation_classes = [ "dann", "mmd", "semantic", "sinkhorn" ]
        
        # loss function
        self.loss = nn.BCEWithLogitsLoss()
        if self.da_loss_fnc_name == "dann":
           self.loss_dann = nn.CrossEntropyLoss()  
           loss_dann = torch.tensor(0.0)  

        self.source_temp = nn.Parameter(torch.tensor(0.))
        if self.da_loss_fnc_name in self.domain_adaptation_classes: 
           print("turning on the temperature for target and DA")
           self.target_temp = nn.Parameter(torch.tensor(0.))
           self.da_temp = nn.Parameter(torch.tensor(0.))

        # metrics
        self.classes = ("noise", "signal")
        metric_args = {"task": "binary"}
        
        self.source_recall = tm.Recall(**metric_args)
        self.source_precision = tm.Precision(**metric_args)
        self.cm_logger = ConfusionMatrixLogger(self.classes)
        self.source_cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
        self.source_cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)

        if self.da_loss_fnc_name in self.domain_adaptation_classes:
            self.target_recall = tm.Recall(**metric_args)
            self.target_precision = tm.Precision(**metric_args)
            self.target_cm_recall = tm.ConfusionMatrix(normalize="true", **metric_args)
            self.target_cm_precision = tm.ConfusionMatrix(normalize="pred", **metric_args)
            # self.embeddings = CombinedEmbeddingPlot(method="umap")


        # network
        self.net = nn.Linear(hit_features, 1)

        # Domain classifier network for DANN
        if self.da_loss_fnc_name == "dann":
           # self.domain_net = nn.Linear(in_features=hit_features,
           #                             out_features=2)
           self.domain_net = nn.Sequential(
                                  nn.Linear(in_features=hit_features, out_features=64),
                                  nn.ReLU(),
                                  nn.Linear(in_features=64,out_features=2)
           )  
           self.domain_classes = ['source', 'target']

    def forward(self, data: Data | list[Data], stage: str = None) -> dict[str, Any]:
        """
        NuGraph3 filter decoder forward pass

        Args:
            data: Graph data object
            stage: Stage name (train/val/test)
        """

        if not self.da_loss_fnc_name in self.domain_adaptation_classes and len(data)==1:
           source_data = data
        else:
           source_data, target_data = data 
            
        def _run_net_and_add_output(data: Any, net: torch.nn.Module, key: str):
            data[key].x_filter = net(data[key].x)
            if isinstance(data, Batch):
               data._slice_dict[key]["x_filter"] = data[key].ptr
               inc = torch.zeros(data.num_graphs, device=data[key].x.device)
               data._inc_dict[key]["x_filter"] = inc

        _run_net_and_add_output(source_data, self.net, "hit")

        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           _run_net_and_add_output(target_data, self.net, "hit")

        loss = loss_source = loss_target = lossDA = raw_lossDA = dann_loss = 0

        x_source = source_data["hit"].x_filter.squeeze(dim=-1) # w/o squeeze: ValueError: Target size (torch.Size([102085])) must be the same as input size (torch.Size([102085, 1]))

        y_source = (source_data["hit"].y_semantic != -1).float()
        w_source = 2 * (-1 * self.source_temp).exp()
        loss_source = w_source * self.loss(x_source, y_source) + self.source_temp

        if self.da_loss_fnc_name in self.domain_adaptation_classes:
           x_target = target_data["hit"].x_filter.squeeze(dim=-1)
           y_target = (target_data["hit"].y_semantic != -1).float()
           w_target = 2 * (-1 * self.target_temp).exp()
           loss_target = w_target * self.loss(x_target, y_target) + self.target_temp

        if self.da_loss_fnc_name == None:
           loss = loss_source

        if self.use_domain_adaptation:
           wDA = 2 * (-1 * self.da_temp).exp()
            
           if self.da_loss_fnc_name == "dann":    
              alpha = 1  
              reversed_S = ReverseLayerF.apply(source_data["hit"].x, alpha)
              xSS = self.domain_net(reversed_S)
              ySS = torch.zeros(xSS.shape[0], dtype=torch.long, device=xSS.device)
            
              reversed_T = ReverseLayerF.apply(target_data["hit"].x, alpha)
              xTT = self.domain_net(reversed_T) 
              yTT = torch.ones(xTT.shape[0], dtype=torch.long, device=xTT.device)    

              combined_image = torch.cat((xSS, xTT), dim=0)  
              combined_label = torch.cat((ySS, yTT), dim=0)
               
              dann_loss = self.loss_dann(combined_image, combined_label)
              raw_lossDA = wDA * dann_loss + self.da_temp

           else:
              sys.exit( f"The function {self.da_loss_fnc_name} does not exist." )

        lossDA = raw_lossDA
        if self.assume_target_label_known: 
           # Total loss
           loss = loss_source + loss_target + lossDA
        else:
           # print("loss = loss_source + lossDA") 
           loss = loss_source + lossDA

        # # calculate metrics
        metrics = {}
        name = "_source/" if self.da_loss_fnc_name else "/"

        if stage:
            metrics[f"loss_filter_{name}{stage}"] = loss
            metrics[f"filter_recall{name}{stage}"] = self.source_recall(x_source, y_source)
            metrics[f"filter_precision{name}{stage}"] = self.source_precision(x_source, y_source)
            if self.da_loss_fnc_name: 
                metrics[f"recall_filter_target/{stage}"] = self.target_recall(x_target, y_target)
                metrics[f"precision_filter_target/{stage}"] = self.target_precision(x_target, y_target)
                # metrics[f"loss_filter_source/{stage}"] = loss_source
                metrics[f"loss_filter_target/{stage}"] = loss_target
                if self.da_loss_fnc_name == "dann":
                    metrics[f"DA_loss_capped_filter/{stage}"] = lossDA
                    metrics[f"DA_loss_temp_scaled_filter/{stage}"] = raw_lossDA 
                    metrics[f"DA_loss_filter/{stage}"] = dann_loss
            
        if stage == "train":
            metrics["temperature/filter"] = self.source_temp
            if self.da_loss_fnc_name in self.domain_adaptation_classes:
                metrics["temperature/filter_target"] = self.target_temp
                metrics["temperature/filter_DA"] = self.da_temp
                
        if stage in ["val", "test"]:
            self.source_cm_recall.update(x_source, y_source)
            self.source_cm_precision.update(x_source, y_source)
            if self.da_loss_fnc_name in self.domain_adaptation_classes:
                self.target_cm_recall.update(x_target, y_target)
                self.target_cm_precision.update(x_target, y_target)
                # self.embeddings.update(source_data["hit"].x, y_source, target_data["hit"].x, y_target)       

        # run network and add output to graph object
        source_data["hit"].x_filter = source_data["hit"].x_filter.sigmoid()
        if self.da_loss_fnc_name in self.domain_adaptation_classes:
            target_data["hit"].x_filter = target_data["hit"].x_filter.sigmoid()
        if isinstance(data, Batch):
            # pylint: disable=protected-access
            data._slice_dict["hit"]["x_filter"] = data["hit"].ptr
            inc = torch.zeros(data.num_graphs, device=data["hit"].x.device)
            data._inc_dict["hit"]["x_filter"] = inc

        return loss, metrics

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        """
        Draw a confusion matrix

        Args:
            cm: Confusion matrix object
        """
        confusion = cm.compute().cpu()
        fig = plt.figure(figsize=[8,6])
        sn.heatmap(confusion,
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel("Assigned label")
        plt.ylabel("True label")
        return fig


    
    def on_epoch_end(self, logger: Logger | list[Logger], stage: str,
                     epoch: int) -> None: # pylint: disable=unused-argument
        """
        NuGraph3 decoder end-of-epoch callback function

        Args:
            logger: PyTorch Lightning logger object(s)
            stage: Training stage
            epoch: Training epoch index
        """
        if not logger:
            return

        if self.da_loss_fnc_name == None:
            self.cm_logger.log(f"filter/recall-matrix-{stage}",
                               self.source_cm_recall, logger, epoch)
            self.cm_logger.log(f"filter/precision-matrix-{stage}",
                               self.source_cm_precision, logger, epoch)
        else:
            logger.experiment.add_figure(f"recall_filter_matrix_source/{stage}",
                                     self.draw_confusion_matrix(self.source_cm_recall),
                                     global_step=epoch)
            self.source_cm_recall.reset()
            
            logger.experiment.add_figure(f"recall_filter_matrix_target/{stage}",
                                     self.draw_confusion_matrix(self.target_cm_recall),
                                     global_step=epoch)
            self.target_cm_recall.reset()
            
            logger.experiment.add_figure(f"precision_filter_matrix_source/{stage}",
                                self.draw_confusion_matrix(self.source_cm_precision),
                                global_step=epoch)
            self.source_cm_precision.reset()
            
            logger.experiment.add_figure(f"precision_filter_matrix_target/{stage}",
                                self.draw_confusion_matrix(self.target_cm_precision),
                                global_step=epoch)
            self.target_cm_precision.reset()
            
            # # # Plot the embedding space 
            # dat1, lab1, dat2, lab2 = self.embeddings.compute()
            # dat1sub, lab1sub = self.embeddings.subsample(dat1, lab1, max_samples=1000)
            # dat2sub, lab2sub = self.embeddings.subsample(dat2, lab2, max_samples=1000)
            
            # embeddings_fig = self.embeddings.plot_combined(dat1sub, lab1sub, dat2sub, lab2sub, epoch=epoch, class_names=self.classes)
            # logger.experiment.add_figure(f"Embeddings filter/{stage}",
            #                          embeddings_fig, global_step=epoch)
            # self.embeddings.reset()