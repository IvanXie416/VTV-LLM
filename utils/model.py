import torch
from torch import nn
from utils.constants import *
from utils.modeling.modeling_pretrain import (
    PretrainVisionTransformer,
    pretrain_videomae_base_patch16_224
)
import os


class VideoMAETactileEncoder(nn.Module):
    def __init__(self):
        super(VideoMAETactileEncoder, self).__init__()
        self.model = pretrain_videomae_base_patch16_224()
        self.output_dim = 768  

    def load_pretrained_weights(self, checkpoint_path):
        
        if not os.path.exists(checkpoint_path):
            print(f"checkpoint not found: {checkpoint_path}")
            return
        
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'module' in state_dict:
            state_dict = state_dict['module']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        new_state_dict = {}
        
        for k, v in state_dict.items():
            if k.startswith('blocks.'):
                new_key = f"encoder.{k}"
                new_state_dict[new_key] = v
            elif k.startswith('patch_embed.'):
                new_key = f"encoder.{k}"
                new_state_dict[new_key] = v
            elif k.startswith('norm.'):
                new_key = f"encoder.{k}"
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        

        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)

        
        return missing_keys, unexpected_keys

    def forward(self, tactile_pixel_values):
        b, l, c, h, w = tactile_pixel_values.shape

        x = tactile_pixel_values.permute(0, 2, 1, 3, 4)
        
        x = x.to(dtype=torch.bfloat16)
        num_patches = self.model.encoder.patch_embed.num_patches
        mask = torch.zeros((b, num_patches), dtype=torch.bool, device=x.device)
        x_vis = self.model.encoder.forward_features(x, mask)
        video_features = x_vis[:, 0]
        
        return video_features
    

        

def sinusoidal_positional_embedding(token_sequence_size, indices, token_embedding_dim, batch_size, n=10000.0):

    positions = indices.float().unsqueeze(-1)
    embeddings = torch.zeros(batch_size, token_sequence_size, token_embedding_dim, device=indices.device)
    denominators = torch.pow(n, 2 * torch.arange(0, token_embedding_dim // 2, device=indices.device) / token_embedding_dim)

    calculated_embeds_sin = torch.sin(positions / denominators)
    calculated_embeds_cos = torch.cos(positions / denominators)

    embeddings[:, :, 0::2] = calculated_embeds_sin
    embeddings[:, :, 1::2] = calculated_embeds_cos

    return embeddings

    
class MultimodalLLMForCausalLM(nn.Module):
    def __init__(self, tokenizer, videomae_model_name, encoder_output_size, cutoff_len, llm, use_vqvae, device):
        super(MultimodalLLMForCausalLM, self).__init__()
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.use_vqvae = use_vqvae
        self.device = device
        self.llm_embedding_size = llm.model.embed_tokens.weight.shape[1]
        self.encoder = VideoMAETactileEncoder(videomae_model_name=videomae_model_name)
        self.encoder_output_size = self.encoder.output_dim
        self.project = nn.Sequential(
            nn.Linear(self.encoder_output_size, self.llm_embedding_size),
            nn.GELU(),
            nn.Linear(self.llm_embedding_size, self.llm_embedding_size),
        )

    def get_dummy_token(self, answer_tokens, question_embeds_len):
        batch_size = answer_tokens.shape[0]
        answer_embeds_len = answer_tokens.shape[1]
        pre_label_token = torch.full((batch_size, question_embeds_len), fill_value=-100, dtype=torch.int64, device=self.device)
        post_padding_len = self.cutoff_len - (question_embeds_len + answer_embeds_len)
        if post_padding_len < 0: post_padding_len = 0
        post_label_token = torch.full((batch_size, post_padding_len), fill_value=-100, dtype=torch.int64, device=self.device)
        return pre_label_token, post_label_token

    def forward(self, question, tactile_pixel_values, answer_tokens, all_indices, images=None):
        answer_tokens = answer_tokens.to(self.device)

        question_embeds = []
        video_token_count = 0
        for chunk in question:
            chunk = chunk[0]
            if "<video>" in chunk:
                current_video_pixels = tactile_pixel_values[:, video_token_count, :, :, :, :].to(self.device)
                idx = all_indices[:, video_token_count, :].to(self.device)
                visual_embeds = self.encoder(current_video_pixels)
                l = 16
                visual_embeds_repeated = visual_embeds.unsqueeze(1).repeat(1, l, 1)
                sinusoidal_embeds = sinusoidal_positional_embedding(
                    token_sequence_size=l,
                    indices=idx,
                    token_embedding_dim=self.encoder_output_size,
                    batch_size=visual_embeds_repeated.shape[0]
                ).to(device=visual_embeds_repeated.device, dtype=torch.bfloat16)
                chunk_embeds = self.project(visual_embeds_repeated + sinusoidal_embeds)
                video_token_count += 1
            else:
                token_ids = torch.tensor(self.tokenizer.encode(chunk), dtype=torch.int64)[1:].to(self.device)
                chunk_embeds = self.llm.get_input_embeddings()(token_ids)
                chunk_embeds = torch.unsqueeze(chunk_embeds, dim=0)
            question_embeds.append(chunk_embeds)
        question_embeds = torch.cat(question_embeds, dim=1)
        answer_embeds = self.llm.get_input_embeddings()(answer_tokens)
        full_embeds_len = question_embeds.shape[1] + answer_embeds.shape[1]
        question_embeds_len = question_embeds.shape[1]
        batch_size = question_embeds.shape[0]
        padding_len = self.cutoff_len - full_embeds_len
        if padding_len > 0:
            padding_embeds = self.llm.get_input_embeddings()(torch.zeros(batch_size, padding_len, device=self.device, dtype=torch.int64))
            input_embeds = torch.cat((question_embeds, answer_embeds, padding_embeds), dim=1)
            attention_mask = torch.cat((torch.ones(batch_size, full_embeds_len, device=self.device), torch.zeros(batch_size, padding_len, device=self.device)), dim=1)
        else:
            input_embeds = torch.cat((question_embeds, answer_embeds), dim=1)[:, :self.cutoff_len, :]
            attention_mask = torch.ones(batch_size, self.cutoff_len, device=self.device)
            available_answer_len = self.cutoff_len - question_embeds_len
            if available_answer_len < 0:
                answer_tokens = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
                question_embeds_len = self.cutoff_len
            elif available_answer_len < answer_tokens.shape[1]:
                answer_tokens = answer_tokens[:, :available_answer_len]
        pre_label_dummy_token, post_label_dummy_token = self.get_dummy_token(answer_tokens, question_embeds_len)
        labels = torch.cat((pre_label_dummy_token, answer_tokens, post_label_dummy_token), dim=1)

        seq_length = input_embeds.shape[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_embeds.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length) 

        out = self.llm(
            inputs_embeds=input_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids 
        )
        return out, question_embeds