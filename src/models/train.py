import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time

    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, criterion, optimizer, device='cuda'):
    soft = nn.Softmax(-1)
    model.train()

    epoch_loss, accuracy, f1, recall, precision = 0, 0, 0, 0, 0

    for i, batch in enumerate(iterator):
        inputs = batch['input_notes'].to(device)
        labels = batch['targets']

        optimizer.zero_grad()

        output = model(inputs)

        loss = criterion(output.cpu(), labels.long())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # calculate metrics
        result = output.argmax(-1).tolist()
        labels = labels.tolist()

        accuracy += accuracy_score(labels, result)
        f1 += f1_score(labels, result, average='micro')
        recall += recall_score(labels, result, average='micro')
        precision += precision_score(labels, result, average='micro')

    accuracy /= ( i +1)
    f1 /= ( i +1)
    epoch_loss /= ( i +1)
    recall /= ( i +1)
    precision /= ( i +1)

    return epoch_loss, accuracy, f1, recall, precision


def beam_search_decoder(predictions, top_k=5):
    output_sequences = [([], 0)]
    for token_probs in predictions:
        new_sequences = []
        for old_seq, old_score in output_sequences:
            for char_index in range(len(token_probs)):
                new_score = old_score + np.log(token_probs[char_index].detach().numpy())
                new_sequences.append((old_seq + [char_index], new_score))

        output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)
        output_sequences = output_sequences[:top_k]

    return output_sequences
