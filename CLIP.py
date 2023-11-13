import torch
from torch import nn
import torch.nn.functional as F

import config as CFG
from modules import ImageEncoder, TextEncoder, ProjectionHead

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

text_subnets = [{'w': 0.25, 'd': 0.5},
                {'w': 0.25, 'd': 0.75},
                {'w': 0.25, 'd': 1.0},
                {'w': 0.5, 'd': 0.5},
                {'w': 0.5, 'd': 0.75},
                {'w': 0.5, 'd': 1.0},
                {'w': 0.75, 'd': 0.5},
                {'w': 0.75, 'd': 0.75},
                {'w': 0.75, 'd': 1.0},
                {'w': 1.0, 'd': 0.5},
                {'w': 1.0, 'd': 0.75},
                {'w': 1.0, 'd': 1.0}]

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # print(image_features.shape)
        # print(text_features.shape)

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        
        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    
    def change_image_encoder_subnet(self, subnet_no):
        self.image_encoder.ofa_network.set_active_subnet(w=image_subnets[subnet_no]['w'],
                                        e=image_subnets[subnet_no]['e'], 
                                        d=image_subnets[subnet_no]['d'])
        manual_subnet = self.image_encoder.ofa_network.get_active_subnet(preserve_weight=True)
        self.image_encoder.model[0] = manual_subnet

    def change_text_encoder_subnet(self, subnet_no):
        self.text_encoder.model.apply(lambda m: setattr(m, 'depth_mult', text_subnets[subnet_no]['d']))
        self.text_encoder.model.apply(lambda m: setattr(m, 'width_mult', text_subnets[subnet_no]['w']))


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    input_ids = torch.randint(5, 300, size=(8, 25))
    attention_mask = torch.ones(8, 25)
    batch = {
        'image': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }

    CLIP = CLIPModel()
    loss = CLIP(batch)
    print("")