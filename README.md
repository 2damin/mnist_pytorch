# mnist_pytorch

MNIST using PyTorch

Model : Lenet5

## Dependency

```
python >= 3.6

Numpy

torch >= 1.9

torchvision >= 0.10

tensorboard

tensorboardX

torchsummary

matplotlib

```

## Run

Training,

```{r, engine='bash', count_lines}
python main.py --mode train --download 1 --output_dir ${output_directory}
```

Evaluation,

```{r, engine='bash', count_lines}
python main.py --mode eval --download 1 --checkpoint ${trained_model} --output_dir ${output_directory}
```

Test,

```{r, engine='bash', count_lines}
python main.py --mode test --download 1 --checkpoint ${trained_model} --output_dir ${output_directory}
```
## Tensorboard

### Visualize training graph

Using Tensorboard,

```{r, engine='bash', count_lines}
tensorboard --logdir= ${output_directory} --port 8888
```

And enter this site "http://localhost:8888" in Chrome/Edge/Firefox/etc
