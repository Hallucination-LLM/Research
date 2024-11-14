import os
import pickle
import numpy as np

class AttentionLoader:
    def __init__(self, directory, aggregation='mean'):

        self.directory = directory
        self.aggregation = aggregation
        self.attentions = None

    def get_reduced_attention(self):

        attensions_agg = []

        for filename in os.listdir(self.directory):

            if filename.endswith('.pkl'):

                with open(os.path.join(self.directory, filename), 'rb') as f:
                    print(f"Processing {filename}")
                    attensions_agg.append(self._reduce_and_aggregate(pickle.load(f)))
                    break
        
        print("All attentions processed.")
        self.attentions = attensions_agg
        return attensions_agg

    def _reduce_and_aggregate(self, att):

        agg_func = self._get_agg_func(att)

        n_generated_tokens = len(att)
        n_layers = len(att[0])
        n_heads, n_prompt_tokens, _ = att[0][0].shape

        num_all_tokens = n_prompt_tokens + n_generated_tokens
        reduced_att = []

        # final_attension = np.zeros(
        #     shape=(n_layers, n_heads, num_all_tokens, num_all_tokens),
        # )

        final_attension = None
        stacked_att = [
            np.stack(token_att, axis=0)[..., :num_all_tokens]
            if i == 0
            else np.stack(token_att, axis=0)[..., :num_all_tokens][:, :, np.newaxis, :]
            for i, token_att in enumerate(att)
        ]

        final_attension = np.concatenate(stacked_att, axis=-2)
        return final_attension

        for i, token_att in enumerate(att):

            aggregated_layer = []
            token_att = np.stack(token_att, axis=0)[..., :num_all_tokens]

            if i == 0:
                # final_attension[:, :, :n_prompt_tokens, :num_all_tokens] = token_att
                final_attension = token_att
            else:
                # token_att = np.stack(token_att, axis=0)[..., :num_all_tokens]
                final_attension[:, :, n_prompt_tokens + i - 1] = token_att


            # for layer_att in token_att:

            #     if i == 0:
            #         layer_att = layer_att[:, :num_prompt_tokens, :num_prompt_tokens]
            #     else:
            #         layer_att = layer_att[:, :num_prompt_tokens + i]

            #     aggregated_layer.append(agg_func(layer_att))

        return final_attension

        reduced_att.append(tuple(aggregated_layer))

        return tuple(reduced_att)
    
    def _get_agg_func(self, x):

        agg_funcs = {
            'mean': np.mean,
            'max': np.max,
            'median': np.median
        }

        return lambda x: agg_funcs[self.aggregation](x, axis=0)

# Example usage:
# loader = AttentionLoader('/path/to/pickled/files', aggregation='mean')
# reduced_attention = loader.get_reduced_attention()
