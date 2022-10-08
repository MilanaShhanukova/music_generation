import numpy as np
import torch


def generate(dataset, model, predict_len):
    all_probabilities = []
    sample_index = np.random.randint(len(dataset))
    sample, target = dataset[0]

    input = sample.unsqueeze(0)
    predictions = sample.tolist()
    for p in range(predict_len):

        output = model(input)
        probabilites = torch.softmax(output, dim=1)
        all_probabilities.append(probabilites)

        predicted_id = torch.argmax(probabilites)
        predictions.append(predicted_id.item())

        input = torch.tensor(predictions[p + 1:], dtype=torch.long).unsqueeze(0)
    return predictions, torch.cat(all_probabilities, 0)
