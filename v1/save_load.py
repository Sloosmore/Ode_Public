import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Assuming you have your TSN_model class defined as before

def save_model(model, optimizer, epoch, loss, filepath):
    """
    Save the model state, optimizer state, and other training info.
    
    Args:
    model (nn.Module): The model to save
    optimizer (torch.optim.Optimizer): The optimizer used in training
    epoch (int): The current epoch number
    loss (float): The current loss value
    filepath (str): Path to save the checkpoint
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
    }, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath, model, optimizer=None, device='cpu'):
    """
    Load a saved model and optionally optimizer state.
    
    Args:
    filepath (str): Path to the saved checkpoint
    model (nn.Module): An instance of the model architecture
    optimizer (torch.optim.Optimizer, optional): The optimizer to load state into
    device (str): The device to load the model onto
    
    Returns:
    model (nn.Module): The loaded model
    optimizer (torch.optim.Optimizer): The loaded optimizer (if provided)
    epoch (int): The epoch at which the model was saved
    loss (float): The loss at which the model was saved
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Model loaded from {filepath}")
    return model, optimizer, epoch, loss

def save_encoder(model, path):
    """Save the encoder part of the model."""
    torch.save({
        'chunk_encoder': model.chunk_encoder.state_dict(),
        'prompt_encoder': model.prompt_encoder.state_dict() if model.chunk_encoder != model.prompt_encoder else None
    }, path)

def save_post_encoder(model, path):
    """Save the post-encoder part of the model."""
    torch.save({
        'pred_layer': model.pred_layer.state_dict()
    }, path)


def load_encoder(model, path):
    """Load the encoder part of the model."""
    checkpoint = torch.load(path)
    model.chunk_encoder.load_state_dict(checkpoint['chunk_encoder'])
    if checkpoint['prompt_encoder'] is not None:
        model.prompt_encoder.load_state_dict(checkpoint['prompt_encoder'])
    else:
        model.prompt_encoder = model.chunk_encoder
    return model 

def load_post_encoder(model, path):
    """Load the post-encoder part of the model."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['pred_layer'])
    return model