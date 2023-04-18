# Datati: Modern (tabular) datasets require modern solutions

> :warning: Warning, alpha version, everything brakes, only tested on Huggingface and single-output datasets for
> now. :warning:


__Dataset to model, in one go!__
**datati** is a small library to streamline tabular dataset loading and preprocessing.
The goal of this library is to minimize the boring boilerplate code that separates choosing a dataset to work on,
and actually getting it ready to train a classification model .
`datati` provides simple interfaces to load, preprocess, and encode a dataset for training your model of choice:
```python
from datati.dataset import Dataset
from datati.models.trees import ContinuousTreeModeler

# load dataset
dataset = Dataset("mstz/adult", config="income", split="train", load_from="huggingface",
                  target_feature="over_threshold")
tree_dataset = ContinuousTreeModeler().process(dataset).to_array()

x, y = tree_dataset[:, :-1], tree_dataset[:, -1]
```
This snippet allows us to load a dataset (`Dataset("mstz/adult")`) in a desired configuration 
(`config="income"`) and split (`split="train""`) of choice.
Then we use a `Modeler` object to map the initial `Dataset` into an encoding suitable for Decision Tree induction
(`modeler.process(dataset)`).

`datati` builds on top of the huggingface hub, providing an interface to integrate it with common preprocessing
pipelines.

# Quickstart
`datati` is currently available on test-pypi:
```shell
pip install --extra-index-url https://test.pypi.org/simple/ datati==0.0.20
```

## What datasets are available?
`datati` allows you to load huggingface (`load_from="huggingface""`), or local (`load_from="local""`) datasets,
whether they are `numpy.array`s, `pandas.DataFrame`s, or `pyarrow.ArrowDataset`s.

I'm curating a list of datasets directly on Huggingface at [huggingface.co/mstz](https://huggingface.co/mstz).
Most are still to be updated (especially dataset cards).

# What can I do with a dataset?
Most operations have no side-effects, that is, they yield a __new__ `Dataset` object, rather than modifying the 
existing one.
Extending `pandas.DataFrame`, all operations supported on a `pandas.DataFrame` are also supported on
`Dataset` instances. Methods yielding a `pandas.DataFrame` have been overwritten to yield a `Dataset` instead.

**Dunders**
`Dataset` implements most dunder methods.
A dataset `d` can be both copied (`copy.copy(d)`) and deepcopied (`copy.deepcopy(d)`), it can be checked for equality,
and hashed (`hash(d)`).

**Conversion to/from other formats**
`Dataset`s can be directly exported to:
- `pandas.DataFrame` (`dataset.to_pandas()`)
- `numpy.array` (`dataset.to_array()`)
- `list` (`dataset.to_list()`)

## Model-specific encoding
> **NOTE** As of now `datati` is aimed exclusively at single-output tabular classifiers, hence string/object features
> are treated as categorical. 

The `Modeler` class (`datati.models.Modeler`) implements a minimal interface to map a dataset for processing for the
algorithm of choice.
Currently, `datati` implements:

|                         | **Algorithm**     | **Info**`                                                  |
|-------------------------|-------------------|------------------------------------------------------------|
| `ContinuousTreeModeler` | **Decision tree** | Categorical features are encoded through target encoding.  |
| `OneHotTreeModeler`     | **Decision tree** | Categorical features are encoded through one-hot encoding. |
| `SBRLModeler`           | **SBRL**          | All features are binned, then binarized.                   |
| `CorelsModeler`         | **CORELS**        | All features are binned, then binarized.                   |

All implemented `Modeler`s leave a trace of their own transformations by enriching the transformed `Dataset` with
transformation-specific mappings:
```python
from pprint import pprint

dataset = Dataset("mstz/adult", config="income", split="train", load_from="huggingface")

# preprocess dataset for decision tree classification
dataset.target_feature = "over_threshold"
modeler = ContinuousTreeModeler()
tree_dataset = modeler.process(dataset)

pprint(tree_dataset.bins_encoding_dictionaries)
# {}
pprint(tree_dataset.one_hot_encoding_dictionaries)
# {}
pprint(tree_dataset.target_encoding_dictionaries)
 #{'marital_status': {'Divorced': array([0.12109962]),
 #                    'Married-AF-spouse': array([0.00057328]),
 #                    'Married-civ-spouse': array([0.25382872]),
 #                    'Married-spouse-absent': array([0.01165679]),
 #                    'Never-married': array([0.3155797]),
 #                    'Separated': array([0.02942863]),
 #                    'Widowed': array([0.02855505])},
 # 'native_country': {'?': array([0.01321285]),
 #                    'Cambodia': array([0.00035489]),
 #                    'Canada': array([0.00232044]),
 #                    'China': array([0.00174715]),
 #                    'Columbia': array([0.00185635]),
 #                    'Cuba': array([0.00204745]),
 # ...
```
Similarly, when applying one-hot encoding, `dataset.one_hot_encoding_dictionary` will hold the encoding indexes:
```python
from pprint import pprint

dataset = Dataset("mstz/adult", config="income", split="train", load_from="huggingface")

# preprocess dataset for decision tree classification
dataset.target_feature = "over_threshold"
modeler = ContinuousTreeModeler()
tree_dataset = modeler.process(dataset)

pprint(tree_dataset.one_hot_encoding_dictionary)
# {'marital_status': {'Divorced': 0,
#                     'Married-AF-spouse': 1,
#                     'Married-civ-spouse': 2,
#                     'Married-spouse-absent': 3,
#                     'Never-married': 4,
#                     'Separated': 5,
#                     'Widowed': 6},
#  'native_country': {'?': 0,
#                     'Cambodia': 1,
#                     'Canada': 2,
#                     'China': 3,
#                     'Columbia': 4,
#                     'Cuba': 5,
# ...

```


### Encoding Legos
Some lower-level modelers can be composed as small building blocks to provide the desired result.
Currently, `datati` implements:

|                    | **Info**                                                                                |
|--------------------|-----------------------------------------------------------------------------------------|
| `TargetModeler`    | Categorical features are encoded through target encoding.                               |
| `OneHotModeler`    | Categorical features are encoded through one-hot encoding.                              |
| `BinModeler`       | Numerical features are discretized into bins.                                           |
| `BinaryModeler`    | **All** features are first binned, then each bin is transformed into a boolean feature. |
| `BoolToIntModeler` | Booleans are mapped to `{0, 1}`.                                                        |
| `IntToBoolModeler` | Integers are mapped to booleans.                                                        |
| `SBRLModeler`      | All features are binned, then binarized.                                                |
| `CorelsModeler`    | All features are binned, then binarized.                                                |


Similarly to `scikit-learn` pipelines, you can implement your own `Modeler` by combining existing modelers through a
pipeline, as it is done in the `ContinuousTreeModeler`.
Here, we first target-encode categorical variables, then transform booleans into integers:
```python
class ContinuousTreeModeler(NumericModeler):
    """Model data as continuous, mapping boolean features to 0/1, and categorical features to target encoders."""
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline(TargetModeler(), BoolToIntModeler(guess_booleans=True))

    def process(self, dataset: Dataset, **kwargs) -> Dataset:
        """Adapt the given `dataset` to be fed to the model of choice.

        Args:
           dataset: The dataset to process.
           **kwargs: Keyword arguments.

        Returns:
           The processed dataset.
        """
        return self.pipeline(dataset, **kwargs)
```
---

# Run on your own (local) dataset
Local or remote doesn't matter, `datati` can be integrated to work on your own local dataset.
All it takes, is to specify that we're working on a local (`local=True`) dataset:
```python
from datati.dataset import Dataset

dataset = Dataset("./adult", load_from="local")
```
