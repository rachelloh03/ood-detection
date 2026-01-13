# OOD Detection

Back to home: [README](../README.md)

## 1. OOD Detector methods and functions in ```main/```

### General Framework

In this project we focus on post-hoc OOD detection, i.e. the model is frozen.

Many post-hoc OOD detection methods follow this pattern:
1. Find a embedding function $h()$ that maps each prompt $x$ to a vector representation $h(x)$. An example of $h(x)$ is the activations at a hidden layer of a model given input $x$.
2. Estimate the distribution of $h(x)$ on a training set.
3. For a new prompt $x'$, compute $h(x')$ and determine if it is an outlier of the distribution via a scoring mechanism. An example is taking the Mahalanobis distance of $h(x')$ to the ID distribution of $h(x)$.
4. Set a threshold on the score and determine if $x'$ is OOD.

A complete example of this OOD detection workflow can be found in `main/example.ipynb`. In the example:
- **Steps 1 and 2** (embedding function and distribution estimation) are performed on the ID training set.
- **Steps 3 and 4** (scoring and thresholding) are applied to both the ID test set and the OOD dataset.

This setup enables evaluation by constructing a confusion matrix comparing OOD predictions against the ground truth. In this context, a **positive** sample is one that is OOD. For example, a sample from the ID test set that is incorrectly predicted as OOD is a **false positive**.

To explain how the OOD detector class works, we first talk about:
- how to express $h()$ via the Transformations Class
- the scoring mechanism.

### Transformations Class
It is possible that $h()$ needs to be fit to the training data, for instance via sklearn tools such as PCA. Therefore, this function is represented as an object, similar to how sklearn tools operate.

$h()$ can usually be broken down into multiple functions (an example is below.) The `Transformations` object encapsulates a pipeline of transformation functions that are applied sequentially, where the output of each transformation becomes the input to the next.

**Example:** A Transformations object ```transform_obj``` can represent the following pipeline:
```
[extract_layers(...), PCA(n_components=20), StandardScaler()]
```
where ```extract_layers(...)``` is a function that obtains the $l$-th hidden layers from input data.

This can be fit to the input data via ```transform_obj.fit(input_data)```.

When this object is called on new data ```transform_obj(new_data)```, it passes the new data through the pipeline using the same parameters learnt from the input data.

The code for this class is in ```main/transformations.py```.

### Scoring Function
Refer to Step 3 in the General Framework. Given the embeddings of new inputs $h(x')$, we need to compare them against the distribution of $h(x)$, the embeddings of ID data.

A **scoring function** does this comparison and outputs OOD scores, with higher scores meaning that it's more likely to be OOD. More concretely, it takes in
- N ID embeddings (shape (N, D) where D is the embedding dimension)
- M new input embeddings (shape (M, D))

and outputs M scores for the new input embeddings.

An example of a scoring function in ```main/scoring_functions.py``` is the Mahalanobis distance. This assumes that the ID embeddings follows a Gaussian distribution and computes a score based on the sample mean and covariance of ID embeddings.

### OOD Detector Class
```
def __init__(
    self,
    embedding_function: Transformations,
    scoring_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    id_train_data: torch.Tensor,
):
```
Expressing $h()$ via a Transformations class and defining a scoring function allow us to completely specify the OOD detector. The OOD detector takes in $h()$ via ```embedding_function``` and the scoring function (```scoring_function```), fits the embedding function on the ID training data ```id_train_data```

An example of how to use this is in ```example.ipynb```.

## 2. Real-time OOD detection in ```real_time_detection/```




