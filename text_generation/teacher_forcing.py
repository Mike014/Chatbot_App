# Implementazione della tecnica di Teacher Forcing
import numpy as np

def teacher_forcing(model, input_data, target_data, epochs, batch_size):
    for epoch in range(epochs):
        for i in range(0, len(input_data), batch_size):
            input_batch = input_data[i:i+batch_size]
            target_batch = target_data[i:i+batch_size]
            model.train_on_batch([input_batch, target_batch], target_batch)
        print(f'Epoch {epoch+1}/{epochs} completed')







