import torch
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers import T5ForConditionalGeneration
from transformers.models.t5.configuration_t5 import T5Config

from dataset import Sift1mDataset


class T5ForPretrain(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, args):
        super(T5ForPretrain, self).__init__(config)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.args = args
        self.config: T5Config

        # for dimension alignment
        self.output_proj = nn.Sequential(
            nn.Linear(Sift1mDataset.VECTOR_DIM // args.num_subspace, self.config.d_model // 2),
            nn.Linear(self.config.d_model // 2, self.config.d_model),
        )

        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(
        self,
        decoder_inputs_embeds,
        encoder_outputs=None,
        inputs_embeds=None,
        labels=None,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        '''
            Args
                decoder_inputs_embeds:  vecids
            Return
                Seq2SeqLMOutput()
        '''

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            past_key_values=past_key_values,            
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)  # (batch_size, output_len, num_clusters)
        
        loss = None
        if labels is not None:
            loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        if not return_dict:
            return loss

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )
