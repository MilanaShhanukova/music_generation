# credit - https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)

def generate_song(author, song_name, start_lyrics, model, tokenizer, top_k=0, top_p=0.92,
                  temperature=0.7,max_length=512):
    """
    Uses hugging face generate API to get lyrics.
    :param author: name of available authors.
    :param song_name: new name of the song.
    :start_lyrics: desirable start of the song.
    :model: trained model.
    :tokenizer: trained tokenizer with tokens artist, song_name, lyrics.
    :top_k: number of words to choose from for generation.
    :top_p: percent of words to choose from for generation, should be used when top_k=0.
    :temperature: how 'confident' generation should be, less than 1 gives more predictable, more than 1 less predictable.
    :return: generated lyrics as text
    """
    input_ids = tokenizer(f'[s:artist] {author} [e:artist] [s:song_name] {song_name} [e:song_name] [s:lyrics] {start_lyrics}',
                          return_tensors='pt')['input_ids']
    sample_output = model.generate(
                input_ids,
                do_sample=True,
                max_length=max_length,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature)

    generated_lyrics = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    return generated_lyrics
