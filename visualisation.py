import json

from transformers import BertConfig,BertTokenizerFast,BertModel
import torch

model_path = '/media/caizf/UUI/XABSA-model/outputs/mbert-en-fr-acs_mtl-5555-64-sentiment/checkpoint'
tokenizer = BertTokenizerFast.from_pretrained(model_path, do_lower_case=True, add_prefix_space=True,
                                              is_split_into_words=True, truncation=True)
bert_config = BertConfig.from_pretrained(model_path)
bert_config.output_hidden_states = True
bert = BertModel.from_pretrained(model_path, config=bert_config)

# /media/caizf/UUI/XABSA-model/outputs/mbert-en-fr-acs_mtl-4444/checkpoint
# /media/caizf/UUI/XABSA-model/outputs/mbert-en-fr-acs_mtl-1111-32-token/checkpoint

for language in ['en','fr','ru','nl','es']:
    en_data = []
    with open('data/rest/gold-'+language+'-test.txt') as f:
        datas = f.read().strip().split('\n\n')
        for data in datas:
            items = data.split('\n')
            words = [i.split('\t')[0] for i in items]
            en_data.append(' '.join(words))

    embeddings = []
    for num, data in enumerate(en_data):
        print(num)
        ids = tokenizer.encode(data)
        attn_mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)
        input_ids, attn_mask, token_type_ids = map(torch.LongTensor,
                                                   [[ids], [attn_mask], [token_type_ids]])
        bert.eval()
        pooled_output = bert(
            input_ids=input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )['last_hidden_state']
        pooled_output = pooled_output.squeeze(0)
        pooled_output = pooled_output[0, :]
        pooled_output = pooled_output.tolist()
        embeddings.append(pooled_output)

    with open(language+'_test_embedding.json', 'w') as f:
        json.dump(embeddings, f)