import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class TSN_model(nn.Module):
    def __init__(self, 
                 encoder_model: str,
                 context_vector_len: int,
                 end_pred_sequential: nn.Sequential,
                 device:str = 'cpu',
                 encoder_tune: bool = False,
                 same_encoder: bool = True,
                 concat_method: tuple = ("u", "v", "|u-v|")
                 ) -> None:
        super().__init__()
        """
        Initializes the TSN_model class.

        Args:
            encoder_model (str): Path to the pre-trained model for encoding.
            context_vector_len (int): Length of the context vector.
            embeding_construct (nn.Sequential): Sequential layers for embedding construction.
            end_pred_sequential (nn.Sequential): Sequential layers for final prediction.
            device (str): The device of the model.
            encoder_tune (bool): Whether to fine-tune the encoder.
            same_encoder (bool): Whether to use the same encoder for chunks and prompts.
        """
        self.device = device
        if(same_encoder):
            self.chunk_encoder = self.prompt_encoder = AutoModel.from_pretrained(encoder_model)
        else:
            self.prompt_encoder = AutoModel.from_pretrained(encoder_model)
            self.chunk_encoder = AutoModel.from_pretrained(encoder_model)
                
        self.pred_layer = end_pred_sequential
        self.encoder_tune = encoder_tune
        self.set_encoder_trainable(encoder_tune)

        self.operations = {
        'u': lambda u, v: u,
        'v': lambda u, v: v,
        '|u-v|': lambda u, v: torch.abs(u - v),
        'u*v': lambda u, v: u * v
        }
        
        self.concat_method = concat_method

    def set_encoder_trainable(self, trainable: bool):
        """
        Set the requires_grad parameter for the encoder(s) in the TSN_model.
        """
        self.encoder_tune = trainable
        if self.chunk_encoder is self.prompt_encoder:
            # If using the same encoder for both
            self.chunk_encoder.requires_grad_(trainable)
        else:
            # If using separate encoders
            self.chunk_encoder.requires_grad_(trainable)
            self.prompt_encoder.requires_grad_(trainable)
        
        print(f"Encoder(s) trainable status set to: {trainable}")

    def get_encoder_trainable(self) -> bool:
        """
        Get the current trainable status of the encoder(s).
        """
        return self.encoder_tune

    def concat_vectors(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.operations[op](u, v) for op in self.concat_method], dim=-1).to(self.device)
              
    def mean_pooling(self, model_output, attention_mask):
        """
        Applies mean pooling to the model output.

        Args:
            model_output: The output from the model.
            attention_mask: The attention mask.

        Returns:
            torch.Tensor: The pooled output.
        """
        #token_embeddings = model_output[0] First element of model_output contained all token embeddings grabed in embed_reshape
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        return torch.sum(model_output * input_mask_expanded, -2) / torch.clamp(input_mask_expanded.sum(-2), min=1e-9)  
       
    def forward(self, caption_dict, threshold_dict):
        """
        Forward pass of the model.

        Args:
            token_chunks: Tokenized chunks.
            chunk_masks: Attention masks for chunks.
            token_prompts: Tokenized prompts.
            prompt_masks: Attention masks for prompts.

        Returns:
            torch.Tensor: The output logits.
        """
        
        # each caption has one if the mid demimention so works to be fine
        out_chunks = self.chunk_encoder(**caption_dict)
        chunk_embeddings = self.mean_pooling(out_chunks[0], caption_dict['attention_mask'])
        
        out_prompt = self.prompt_encoder(**threshold_dict)
        prompt_embeddings = self.mean_pooling(out_prompt[0], threshold_dict['attention_mask'])

        return self.pred_layer(self.concat_vectors(chunk_embeddings, prompt_embeddings)).squeeze()

if __name__== '__main__':
    MiniLM_L6 = {'path':'sentence-transformers/all-MiniLM-L6-v2', 'size':384}
    model_01 = TSN_model(encoder_model=MiniLM_L6['path'], context_vector_len= MiniLM_L6['size'])