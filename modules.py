import torch
from torch import nn
import timm
from transformers import DistilBertModel, DistilBertConfig, BertForSequenceClassification, BertConfig
import config as CFG
from once_for_all.ofa.model_zoo import ofa_net

image_subnets = [{'w': 1, 'd': 2, 'e': 0.35}, 
           {'w': 1, 'd': 2, 'e': 0.25},
           {'w': 1, 'd': 1, 'e': 0.35},
           {'w': 1, 'd': 0, 'e': 0.25},
           {'w': 0.65, 'd': 2, 'e': 0.35},
           {'w': 0.65, 'd': 1, 'e': 0.2},
           {'w': 0.65, 'd': 1, 'e': 0.25},
           {'w': 0.65, 'd': 0, 'e': 0.35},
           {'w': 0.35, 'd': 2, 'e': 0.2},
           {'w': 0.35, 'd': 1, 'e': 0.25},
           {'w': 0.35, 'd': 0, 'e': 0.35},
           {'w': 0.35, 'd': 0, 'e': 0.2}]

image_config_class = BertConfig
image_config = image_config_class.from_pretrained("pretrained/dynabert/MRPC/", num_labels=768)
image_model = BertForSequenceClassification.from_pretrained("pretrained/dynabert/QQP/", config=image_config, ignore_mismatched_sizes=True)
    
# model = nn.Sequential(
#             ofa_net('ofa_resnet50', pretrained=True),
#             nn.Linear(1000, 2048)
#         )
class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self,  model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        # self.model = timm.create_model(
        #     model_name, pretrained, num_classes=0, global_pool="avg"
        # )
        #agandhi98 
        # print(submodel)
        # print(image_subnets[submodel])
        self.ofa_network = ofa_net('ofa_resnet50', pretrained=True)
        shared_linear_layer = nn.Linear(1000, 2048)
        
        self.ofa_network.set_active_subnet(w=1,
                                      e=0.35, 
                                      d=2)
        manual_subnet = self.ofa_network.get_active_subnet(preserve_weight=True)
        self.model = nn.Sequential(
            manual_subnet,
            shared_linear_layer
        )
        
        print(self.model.eval())
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        # if pretrained:
        #     self.model = DistilBertModel.from_pretrained(model_name)
        # else:
        #     self.model = DistilBertModel(config=DistilBertConfig())
        # image_model.apply()
        self.model = image_model
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        print(self.model.eval())
        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state = output.last_hidden_state
        return output



class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        # print(x[1])
        # print(x)
        # print(x.shape)
    
        try:
            x = x.logits
        except:
            pass
        print(x.shape)
        # x = x.logits
        
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

